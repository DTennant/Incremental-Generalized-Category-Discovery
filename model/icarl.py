import argparse

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import copy
import pandas as pd
from PIL import Image

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model.loss import (
    info_nce_logits,
    SupConLoss,
    DistillLoss,
    ContrastiveLearningViewGenerator,
    get_params_groups,
)
from sklearn.metrics.pairwise import cosine_similarity


class iCarlNet(nn.Module):
    def __init__(self, feature_extractor, dino_head, args=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.dino_head = dino_head
        self.args = args
        self.n_classes = len(args.classes_in_labelled)

        # List containing exemplar_sets
        # each entry is a list of pathes to the images
        """
        [
            [(path1.png, label1), (path2.png, label1), ...],
            [(path1.png, label2), (path2.png, label2), ...],
        ]
        """
        self.exemplar_sets = []

        # Learning method
        self.icarl_cls_loss = nn.CrossEntropyLoss()
        self.dist_loss = nn.BCELoss()

        # Means of exemplars
        self.compute_means = True
        self.exemplar_means = []

    def forward(self, x, return_x=False):
        feature = self.feature_extractor(x)
        output = self.dino_head(feature, return_x=return_x)
        return output

    def incremental_classes(self, n):
        self.n_classes += n

    def classify(self, x, transform):
        batch_size = x.shape[0]

        if self.compute_means:
            self.args.logger.info("Computing means of exemplars")
            exemplar_means = []
            for exemplar_set in self.exemplar_sets:
                features = []
                for ex, _ in exemplar_set:
                    ex = transform(Image.open(ex)).cuda()
                    ex = ex.unsqueeze(0)
                    feature = self.feature_extractor(ex)
                    features.append(feature)
                features = torch.cat(features, dim=0)
                features = features / features.norm(dim=-1, keepdim=True)
                mean = features.mean(dim=0)
                mean = mean / mean.norm(dim=-1, keepdim=True)
                exemplar_means.append(mean)
            self.exemplar_means = exemplar_means
            self.compute_means = False
            self.args.logger.info("Done computing means of exemplars")

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means, dim=0)
        means = torch.stack([means] * batch_size, dim=0)
        means = torch.transpose(means, 1, 2)

        features = self.feature_extractor(x)
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.unsqueeze(2)
        features = features.expand_as(means)

        dist = (features - means).pow(2).sum(dim=1).squeeze()
        _, preds = dist.min(dim=1)
        return preds

    @torch.no_grad()
    def construct_exemplar_sets(self, train_loader, n, args):
        # import ipdb; ipdb.set_trace()
        self.args.logger.info("Constructing exemplar sets")
        self.eval()
        preds, labels, mask_labs, paths = [], [], [], []
        feats = []
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            images, class_labels, uq_idxs, path, mask_lab = batch
            paths.append(path)
            images = images[0].cuda(non_blocking=True)  # only one view of the image
            features = self.feature_extractor(images)
            features = features / features.norm(dim=-1, keepdim=True)
            _, pred = self.dino_head(features)
            preds.append(pred.argmax(1).cpu().numpy())
            labels.append(class_labels.cpu().numpy())
            mask_labs.append(mask_lab.cpu().numpy())
            feats.append(features.cpu().numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        mask_labs = np.concatenate(mask_labs).astype(bool)
        features = np.concatenate(feats)
        paths = np.concatenate(paths)

        labels_used = np.zeros(
            labels.shape
        )  # when mask_lab is True, use labels, when False, use preds
        labels_used[mask_labs] = labels[mask_labs]
        labels_used[~mask_labs] = preds[~mask_labs]
        labels_used = labels_used.astype(np.int32)

        for lbl in np.unique(labels_used):
            feats = features[labels_used == lbl]
            lbl_paths = paths[labels_used == lbl]
            sub_labels_used = labels_used[labels_used == lbl]
            mean_feat = feats.mean(axis=0)  # mean of features
            mean_feat = mean_feat / np.linalg.norm(mean_feat)  # normalize

            exemplar_set = []
            exemplar_features = []
            for k in range(n):
                S = np.sum(exemplar_features, axis=0)
                phi = feats
                mu = mean_feat
                mu_p = 1.0 / (k + 1) * (phi + S)
                mu_p = mu_p / np.linalg.norm(mu_p)
                i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))

                exemplar_set.append(lbl_paths[i])
                exemplar_features.append(feats[i])

            self.exemplar_sets.append([(item, lbl) for item in exemplar_set])

    def reduce_exemplar_sets(self, n):
        for idx, exemplar_set in enumerate(self.exemplar_sets):
            self.exemplar_sets[idx] = exemplar_set[:n]

    def combine_dataset_with_exemplars(self, train_dataset, n, args):
        # return train_dataset + self.exemplar_sets
        # import ipdb; ipdb.set_trace()
        self.reduce_exemplar_sets(n)
        file_names, category_ids = [], []
        for exemplar_set in self.exemplar_sets:
            for ex, cat in exemplar_set:
                file_names.append(ex)
                category_ids.append(cat)
        new_df = pd.concat(
            [
                train_dataset.data,
                pd.DataFrame({"file_name": file_names, "category_id": category_ids}),
            ]
        )
        new_df = new_df.drop_duplicates(
            subset=[
                "file_name",
            ],
            keep="first",
        )
        new_dataset = copy.deepcopy(train_dataset)
        new_dataset.data = new_df
        return new_dataset


def calculate_density_batch(features, K, batch_size=8):
    num_rows = features.shape[0]
    all_densities = []
    all_knn_indices = []

    for i in tqdm(range(0, num_rows, batch_size)):
        end = min(i + batch_size, num_rows)
        batch_features = features[i:end, :]
        sim_matrix = F.cosine_similarity(
            batch_features[None, :, :], features[:, None, :], dim=-1
        ).T
        _, indices = torch.sort(sim_matrix, descending=True, dim=1)
        knn_indices = indices[:, : K + 1]  # +1 to include itself
        knn_similarities = torch.gather(sim_matrix, 1, knn_indices)
        avg_density = knn_similarities[:, 1:].mean(dim=1)  # Exclude itself

        all_densities.append(avg_density.detach().cpu())
        all_knn_indices.append(
            knn_indices[:, 1:].detach().cpu()
        )  # Exclude itself for k-NN list

    densities = torch.cat(all_densities)
    knn_indices = torch.cat(all_knn_indices)

    return densities, knn_indices


def find_density_peaks(knn_indices, densities):
    density_peaks = []
    for i, knn in enumerate(knn_indices):
        if all(densities[i] > densities[knn]):
            density_peaks.append(i)
    return density_peaks


def soft_nearest_neighbor(h_i, density_peaks, labels, tau):
    exp_dot_products = torch.exp(torch.matmul(h_i, density_peaks.T) / tau)
    numerators = torch.matmul(exp_dot_products, labels)
    denominators = exp_dot_products.sum(dim=-1)
    p_i = numerators / denominators.unsqueeze(-1)
    return p_i


class SNNDensityNet(iCarlNet):
    def __init__(self, feature_extractor, dino_head, args=None):
        super().__init__(feature_extractor, dino_head, args=args)
        self.density_peaks = {}  # path: label
        self.K = args.density_k
        self.tau = args.density_tau
        self.transform = None

        self.num_class = 0
        self.cache_pixels = None
        self.cache_labels = None
        self.labeled_peak_paths = set()

    def forward(
        self,
        x,
    ):
        feature = self.feature_extractor(x)
        output = self.classify(feature, use_feature=True)
        proj, dino_logits = self.dino_head(feature)
        return proj, dino_logits, output

    def classify(self, x, transform=None, use_feature=False):
        batch_size = x.shape[0]
        if transform is None:
            transform = self.transform

        if use_feature:
            h_i = x
        else:
            h_i = self.feature_extractor(x)
        h_i = h_i / h_i.norm(dim=-1, keepdim=True)

        if self.cache_pixels is None:
            with torch.no_grad():
                peak_features, labels = [], []
                class ds:
                    def __init__(self, density_peaks, args):
                        self.density_peaks = density_peaks
                        self.args = args
                    def __len__(self):
                        return len(self.density_peaks)
                    def __getitem__(self, idx):
                        path, label = list(self.density_peaks.items())[idx]
                        img = transform(Image.open(os.path.join(self.args.root, path)))#.cuda()
                        return img, label#.cuda()
                dataset = ds(self.density_peaks, self.args)
                dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
                # for path, label in tqdm(self.density_peaks.items()):
                for peak, label in tqdm(dataloader):
                    # peak = transform(Image.open(os.path.join(self.args.root, path))).cuda().unsqueeze(0)
                    peak, label = peak.cuda(), label.cuda()
                    peak_feature = self.feature_extractor(peak)
                    peak_feature = peak_feature / peak_feature.norm(dim=-1, keepdim=True)
                    labels.append(label)
                    peak_features.append(peak_feature)
            self.cache_pixels = torch.cat(peak_features, dim=0)
            labels = torch.cat(labels) 
            self.cache_labels = labels

        peak_features = self.cache_pixels

        labels = self.cache_labels

        labels_one_hot = torch.zeros(peak_features.shape[0], self.num_class).cuda().scatter_(
            1, labels.unsqueeze(-1), 1
        )

        preds = soft_nearest_neighbor(h_i, peak_features, labels_one_hot, self.tau)

        return preds

    def combine_dataset_with_exemplars(self, train_dataset, args):
        # return train_dataset + self.exemplar_sets
        # import ipdb; ipdb.set_trace()
        file_names, category_ids = [], []
        for exemplar_set in self.density_peaks.items():
            for ex, cat in exemplar_set:
                if not (ex in self.labeled_peak_paths):
                    continue
                file_names.append(ex)
                category_ids.append(cat)
        new_df = pd.concat(
            [
                train_dataset.data,
                pd.DataFrame({"file_name": file_names, "category_id": category_ids}),
            ]
        )
        new_df = new_df.drop_duplicates(
            subset=[
                "file_name",
            ],
            keep="first",
        )
        new_dataset = copy.deepcopy(train_dataset)
        new_dataset.data = new_df
        return new_dataset

    @torch.no_grad()
    def construct_exemplar_sets(self, train_loader, n, args):
        # import ipdb; ipdb.set_trace()
        self.args.logger.info("Constructing density peaks")
        self.eval()
        preds, labels, mask_labs, paths = [], [], [], []
        feats = []
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            images, class_labels, uq_idxs, path, mask_lab = batch
            paths.append(path)
            images = images[0].cuda(non_blocking=True)  # only one view of the image
            features = self.feature_extractor(images)
            features = features / features.norm(dim=-1, keepdim=True)
            # _, pred = self.dino_head(features)
            # pred = self.classify(features, use_feature=True)
            # preds.append(pred.argmax(1).cpu().numpy())
            labels.append(class_labels.cpu().numpy())
            mask_labs.append(mask_lab.cpu().numpy())
            feats.append(features.cpu().numpy())

            for p, cl in zip(path, class_labels.cpu().numpy()):
                if p in self.density_peaks.keys():
                    self.density_peaks[p] = cl
                    self.labeled_peak_paths.add(p)

        # preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        mask_labs = np.concatenate(mask_labs).astype(bool)
        features = np.concatenate(feats)
        paths = np.concatenate(paths)

        labels_used = -1 * np.ones(labels.shape)
        labels_used[mask_labs] = labels[mask_labs]
        labels_used = labels_used.astype(np.int32)

        densities, knn_ind = calculate_density_batch(
            torch.from_numpy(features).cuda(), self.K
        )

        self.density_peaks_ind = find_density_peaks(knn_ind, densities)

        num_labelled_class = self.num_class = labels_used.max() + 1
        for ind in self.density_peaks_ind:
            # if label used not -1, then it is a labeled peak, otherwise create a new class for that peak
            if labels_used[ind] != -1:
                self.density_peaks.update({paths[ind]: labels_used[ind]})
                self.labeled_peak_paths.add(paths[ind])
            else:
                # new class
                self.density_peaks.update({paths[ind]: self.num_class})
                self.num_class += 1

        peak_count_labeled = np.zeros(self.num_class)
        for _, label in self.density_peaks.items():
            peak_count_labeled[label] += 1

        # getting peaks for all labeled categories.
        for label in range(num_labelled_class):
            if peak_count_labeled[label] < n:
                indices = np.where(labels_used == label)[0]
                if len(indices) > 0:
                    random_peak_indices = np.random.choice(
                        indices, int(n - peak_count_labeled[label]), replace=False
                    )
                    for ind in random_peak_indices:
                        self.density_peaks.update({paths[ind]: label})
                        self.labeled_peak_paths.add(paths[ind])

        self.args.logger.info("Done")
