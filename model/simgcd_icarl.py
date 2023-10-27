import argparse

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model.loss import info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups

from model.icarl import iCarlNet, SNNDensityNet

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.in_dim = in_dim
        self.out_dim = out_dim
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_x=False):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        if return_x:
            return x_proj, logits, x
        return x_proj, logits
    


def train(student, train_loaders, test_loaders, unlabelled_train_loaders, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train_loaders[-1]) * (args.epochs - args.start_epoch),
            eta_min=args.lr * 1e-3,
        )

    if args.resume:
        load_dict = torch.load(args.resume, map_location='cpu')
        model_dict = load_dict['model']
        optimizer_dict = load_dict['optimizer']
        lr_scheduler_dict = load_dict['lr_scheduler']
        args.start_epoch = load_dict['epoch']
        
        student.load_state_dict(model_dict)
        optimizer.load_state_dict(optimizer_dict)
        exp_lr_scheduler.load_state_dict(lr_scheduler_dict)
        
        args.logger.info(f"Resumed from {args.resume}")

    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    # inductive
    best_test_acc_ubl = 0
    # transductive
    best_train_acc_lab = 0
    best_train_acc_ubl = 0 
    best_train_acc_all = 0

    for epoch in range(args.start_epoch, args.epochs):
        loss_record = AverageMeter()

        student.train()
        student.cache_pixels = None
        for batch_idx, batch in enumerate(train_loaders[-1]):
            images, class_labels, uq_idxs, _, mask_lab = batch
            # mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_dino_out, student_out = student(images)
                teacher_out = student_out.detach()
                teacher_dino_out = student_dino_out.detach()

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # clustering, unsup
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                # DINO loss
                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_dino_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                dino_cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # clustering, unsup
                dino_cluster_loss = cluster_criterion(student_dino_out, teacher_dino_out, epoch)
                avg_probs = (student_dino_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                dino_cluster_loss += args.memax_weight * me_max_loss

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'dino_cls_loss: {dino_cls_loss.item():.4f} '
                pstr += f'dino_cluster_loss: {dino_cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                loss = 0
                loss += args.snn_weight * sigmoid_rampup(epoch, args.rampup_len) \
                    * ((1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss)
                loss += (1 - args.sup_weight) * dino_cluster_loss + args.sup_weight * dino_cls_loss
                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                
            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            # Step schedule
            exp_lr_scheduler.step()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loaders[-1]), loss.item(), pstr))
                
            wandb.log({
                'cls_loss': cls_loss.item(),
                'cluster_loss': cluster_loss.item(),
                'dino_cls_loss': dino_cls_loss.item(),
                'dino_cluster_loss': dino_cluster_loss.item(),
                'sup_con_loss': sup_con_loss.item(),
                'contrastive_loss': contrastive_loss.item(),
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
            })

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        save_dict = {
            'model': student.state_dict(),
            'peaks': student.density_peaks,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': exp_lr_scheduler.state_dict(),
            'epoch': epoch,
        }
        torch.save(save_dict, args.model_path[:-3] + f'_stage_{len(train_loaders)}.pth')
        args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_stage_{len(train_loaders)}.pth'))

        # args.logger.info('Testing on unlabelled examples in the training data...')
        # outputs = test_multi_test_loader(student, unlabelled_train_loaders, epoch=epoch, save_name='Train ACC Unlabelled', args=args)

        # args.logger.info('Testing on disjoint test set...')
        # outputs_test = test_multi_test_loader(student, test_loaders, epoch=epoch, save_name='Test ACC', args=args)

        for stage_idx, (unlabelled_train_loader, test_loader, classes_in_labelled, classes_in_unlabelled, ) in enumerate(zip(unlabelled_train_loaders, test_loaders, 
                                                                                                    args.classes_in_labelled_list, args.classes_in_unlabelled_list, )):
            # NOTE: here we need make sure these classes are in the target transform
            args.classes_in_labelled, args.classes_in_unlabelled = classes_in_labelled, classes_in_unlabelled
            stage_idx += 1

            args.logger.info('Testing on unlabelled examples in the training data...')
            output = test(student, unlabelled_train_loader, epoch=epoch, save_name=f'Train ACC Unlabelled stage {stage_idx}', stage_idx=stage_idx-1, args=args) 

            args.logger.info('Testing on disjoint test set...')
            output_test = test(student, test_loader, epoch=epoch, save_name=f'Test ACC stage {stage_idx}', stage_idx=stage_idx-1, args=args)

            all_acc, old_acc, new_acc = output['All'], output['Old'], output['New']
            all_acc_test, old_acc_test, new_acc_test = output_test['All'], output_test['Old'], output_test['New']
            
            wandb.log({f'stage_{stage_idx}_{len(train_loaders)}_' + k + '_train': v for k, v in output.items()})
            wandb.log({f'stage_{stage_idx}_{len(train_loaders)}_' + k + '_test': v for k, v in output_test.items()})

            args.logger.info('Train Accuracies stage {}: All {:.4f} | Old {:.4f} | New {:.4f}'.format(stage_idx, all_acc, old_acc, new_acc))
            args.logger.info('Test Accuracies stage {}: All {:.4f} | Old {:.4f} | New {:.4f}'.format(stage_idx, all_acc_test, old_acc_test, new_acc_test))

            if stage_idx == len(train_loaders):
                if new_acc_test > best_test_acc_ubl:

                    args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
                    args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

                    torch.save(save_dict, args.model_path[:-3] + f'_{stage_idx}_{len(train_loaders)}_best.pt')
                    args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_{stage_idx}_best.pt'))

                    # inductive
                    best_output_test = output_test
                    best_test_acc_ubl = new_acc_test
                    # transductive            
                    best_output = output
                    best_train_acc_lab = old_acc
                    best_train_acc_ubl = new_acc
                    best_train_acc_all = all_acc

                args.logger.info(f'Exp Name: {args.exp_name} Exp id: {args.exp_id}')
                best_pstr = f'Metrics with best model on train set: '
                output_dict = best_output_test if args.eval_setting == 'inductive' else best_output
                for k, v in output_dict.items():
                    best_pstr += f'{k}: {v:.4f} '
                    wandb.log({f'stage_{stage_idx}_{len(train_loaders)}_best_{k}': v})
                args.logger.info(best_pstr)

def test(model, test_loader, epoch, save_name, stage_idx, args):
    model.eval()
    cur_stage = args.stage - 1

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', ])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    
    parser.add_argument('--n_overlap_cls', type=int, default=-1, help='how many classes overlaps, -1 means all labelled cls are in unlabelled')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)
    parser.add_argument('--eval_setting', type=str, default='transductive', choices=['transductive', 'inductive'], help='options: inductive, transductive')

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--init_estimate_num_cls', type=int, help='the initial estimate of the number of classes, None if use GT', default=None)
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)

    parser.add_argument('--use_small_set', action='store_true', default=False)
    parser.add_argument('--split_crit', type=str, )

    parser.add_argument('--exemplar_number', type=int, default=3)
    parser.add_argument('--snn_weight', type=float, default=0.2)
    parser.add_argument('--density_k', type=int, default=10)
    parser.add_argument('--density_tau', type=float, default=0.1)
    parser.add_argument('--return_path', action='store_true', default=False)
    
    parser.add_argument('--rampup_len', default=50, type=int)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--runner_name', default='simgcd', type=str, )
    parser.add_argument('--entity', default='tennant', type=str, help='entity for wandb')
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--exp_id', default=None, type=str)
    parser.add_argument('--tags', default=None, type=str, help='- separated tags for the experiment; e.g. baseline-ssb')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    # args = get_class_splits(args)

    init_experiment(args, runner_name=[args.runner_name], exp_id=args.exp_id)
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    if args.split_crit in ['year', 'loc_year']:
        args.max_stage = 4
    elif args.split_crit in ['location']:
        args.max_stage = 3
        

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.mlp_out_dim = 10000

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    train_loaders, test_loaders_unlabelled, test_loaders_labelled = [], [], []

    wandb.init(project='IncreNat_gh', entity=args.entity, name=args.exp_name, 
                tags=args.tags.split('-') if args.tags is not None else None, 
                resume=True if args.resume is not None else None)
    wandb.config.update(args)
                
    # get the correct names for args
    args.stage = 0
    args.existing_mapping = None
    args.train_classes = None
    _ = get_datasets(args.dataset_name, None, None, args)

    # Reload the model to be icarl
    # model = iCarlNet(backbone, projector, args).to(device)
    projector = DINOHead(in_dim=768, out_dim=args.mlp_out_dim, nlayers=3)
    model = SNNDensityNet(backbone, projector, args).to(device)
    args.logger.info('model build')
    # __import__("ipdb").set_trace()
    
    args.classes_in_labelled_list = []
    args.classes_in_unlabelled_list = []
    for stage_idx in range(args.max_stage):
        args.stage = stage_idx
        train_dataset, test_dataset, unlabelled_train_examples_test, _ = get_datasets(args.dataset_name,
                                                                                             train_transform,
                                                                                             test_transform,
                                                                                             args)
        model.transform = test_transform
        # update train
        args.target_transform = train_dataset.labelled_dataset.target_transform
        args.classes_in_labelled_list.append(args.classes_in_labelled)
        args.classes_in_unlabelled_list.append(args.classes_in_unlabelled)

        density_train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, 
                                          shuffle=False, pin_memory=True)
        model.construct_exemplar_sets(density_train_loader, args.exemplar_number, args)

        if stage_idx > 0:
            new_labelled_dataset = model.combine_dataset_with_exemplars(train_dataset.labelled_dataset, args)
            new_labelled_dataset.uq_idxs = np.array(range(len(new_labelled_dataset)))
            
            train_dataset.labelled_dataset = new_labelled_dataset

        label_len = len(train_dataset.labelled_dataset)
        unlabelled_len = len(train_dataset.unlabelled_dataset)
        args.logger.info(f'label length: {label_len}, unlabel length: {unlabelled_len}')
        sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

        train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                                  sampler=sampler, drop_last=True, pin_memory=True)
        test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                            batch_size=256, shuffle=False, pin_memory=False)
        test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                          batch_size=256, shuffle=False, pin_memory=False)

        train_loaders.append(train_loader)
        test_loaders_unlabelled.append(test_loader_unlabelled)
        test_loaders_labelled.append(test_loader_labelled)

        train(model, train_loaders, test_loaders_labelled, test_loaders_unlabelled, args)
