from torchvision.datasets import CIFAR10, CIFAR100
from copy import deepcopy
import numpy as np
import torch

from data.data_utils import subsample_instances
from config import cifar_10_root, cifar_100_root


class CustomCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):

        super(CustomCIFAR10, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


class CustomCIFAR100(CIFAR100):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)

class PartialCIFAR100(CIFAR100):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Only returns the data within the target list and select examples based on the partition.

    """
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False,
                 target_list=range(50), partition=(0, 1.0)):

        super(PartialCIFAR100, self).__init__(root, split == 'train', transform, target_transform, download)

        ind = [
            i for i in range(len(self.targets))
            if self.targets[i] in target_list
        ]
        self.target_list = target_list
        self.data = self.data[ind]
        self.targets = np.array(self.targets)
        self.targets = self.targets[ind].tolist()

        self.targets = torch.tensor(self.targets).long()

        data_of_each_class = []
        label_of_each_class = []
        for label in target_list:
            data_of_cur_class = self.data[self.targets == label]
            label_of_cur_class = self.targets[self.targets == label]
            lower, upper = partition
            lower_idx = int(lower * len(data_of_cur_class)) if lower is not None else 0
            upper_idx = int(upper * len(data_of_cur_class)) if upper is not None else len(data_of_cur_class)
            data_of_each_class.append(data_of_cur_class[lower_idx: upper_idx])
            label_of_each_class.append(label_of_cur_class[lower_idx: upper_idx])
        self.data = np.concatenate(data_of_each_class, axis=0)
        self.targets = torch.cat(label_of_each_class, dim=0)
        
        self.uq_idxs = np.array(range(len(self)))
        
    def concat(self, other_dataset):
        self.data = np.concatenate([self.data, other_dataset.data], axis=0)
        self.targets = torch.cat([self.targets, other_dataset.targets], dim=0)
        
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)

def subsample_dataset(dataset, idxs):

    # Allow for setting in which all empty set of indices is passed

    if len(idxs) > 0:

        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]

        return dataset

    else:

        return None


def subsample_dataset_lt(dataset, args=None):
    cls_num = len(np.unique(np.array(dataset.targets)))
    def get_img_num_per_cls(dataset, imb_type='exp', imb_factor=0.02):
        img_max = len(dataset) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls
    
    def gen_imbalanced_data(dataset, img_num_per_cls):
        new_idxs = []
        targets_np = np.array(dataset.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        dataset.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            dataset.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_idxs.append(selec_idx)
        new_idxs = np.concatenate(new_idxs, axis=0)
        return new_idxs

    img_num_per_cls = get_img_num_per_cls(dataset, imb_type='exp', imb_factor=args.imbalance_factor)
    new_idxs = gen_imbalanced_data(dataset, img_num_per_cls)
    dataset = subsample_dataset(dataset, new_idxs)
    return dataset


def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_cifar_10_datasets(train_transform, test_transform, train_classes=(0, 1, 8, 9),
                       prop_train_labels=0.8, split_train_val=False, seed=0, args=None):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Sample the unlabelled data to only have overlap_cls and unlabelled_classes
    if args.n_overlap_cls != -1:
        cls2sample = list(set(args.overlap_cls).union(set(args.unlabeled_classes)))
        train_dataset_unlabelled = subsample_classes(deepcopy(train_dataset_unlabelled), include_classes=cls2sample)

    # Get test set for all classes
    test_dataset = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


def get_cifar_100_datasets(train_transform, test_transform, train_classes=range(80),
                       prop_train_labels=0.8, split_train_val=False, seed=0, args=None):

    np.random.seed(seed)
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])

    # Init entire training set
    whole_training_set = CustomCIFAR100(root=cifar_100_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Sample the unlabelled data to only have overlap_cls and unlabelled_classes
    if args.n_overlap_cls != -1:
        cls2sample = list(set(args.overlap_cls).union(set(args.unlabeled_classes)))
        train_dataset_unlabelled = subsample_classes(deepcopy(train_dataset_unlabelled), include_classes=cls2sample)

    # Get test set for all classes
    test_dataset = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets

def get_cifar_100_lt_datasets(train_transform, test_transform, train_classes=range(80),
                            prop_train_labels=0.8, split_train_val=False, seed=0, args=None):
    np.random.seed(seed)
    # Init entire training set
    whole_training_set = CustomCIFAR100(root=cifar_100_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Sample the unlabelled data to only have overlap_cls and unlabelled_classes
    if args.n_overlap_cls != -1:
        cls2sample = list(set(args.overlap_cls).union(set(args.unlabeled_classes)))
        train_dataset_unlabelled = subsample_classes(deepcopy(train_dataset_unlabelled), include_classes=cls2sample)
        
    # Sample the unlabelled data to be long tail
    train_dataset_unlabelled = subsample_dataset_lt(deepcopy(train_dataset_unlabelled), args)

    # Get test set for all classes
    test_dataset = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


PARTITION_CONFIG_MIX = {
    # (target_list, (partition_lower, partition_upper))
    'stage 1': ((list(range(70)), (0, 0.87)),),
    'stage 2': ((list(range(70)), (0.87, 0.95)),
                (list(range(70, 80)), (0, 0.7)),),
    'stage 3': ((list(range(70)), (0.95, 0.97)),
                (list(range(70, 80)), (0.7, 0.9)),
                (list(range(80, 90)), (0, 0.9)),),
    'stage 4': ((list(range(70)), (0.97, 1.0)),
                (list(range(70, 80)), (0.9, 1.0)),
                (list(range(80, 90)), (0.9, 1.0)),
                (list(range(90, 100)), (0, 1.0)),),
}

def get_cifar_100_icd_dataset(train_transform, test_transform, train_classes=range(80),
                       prop_train_labels=0.8, split_train_val=False, stage=0, seed=0, args=None):
    # this is to return dataset used for each stage of class incremental learning
    # using the above partition config
    # train_classes, prop_train_labels, split_train_val is not used here
    np.random.seed(seed)
    config = PARTITION_CONFIG_MIX[f'stage {stage+1}']
    
    train_d, test_d = None, None
    for target_list, (lower, upper) in config:
        if train_d is None:
            train_d = PartialCIFAR100(root=cifar_100_root, split='train', transform=train_transform, download=True, target_list=target_list, partition=(lower, upper))
            test_d = PartialCIFAR100(root=cifar_100_root, split='test', transform=test_transform, download=True, target_list=target_list, partition=(lower, upper))
        else:
            train_d.concat(PartialCIFAR100(root=cifar_100_root, split='train', transform=train_transform, download=True, target_list=target_list, partition=(lower, upper)))
            test_d.concat(PartialCIFAR100(root=cifar_100_root, split='test', transform=test_transform, download=True, target_list=target_list, partition=(lower, upper)))
            
    return train_d, test_d
    


if __name__ == '__main__':

    x = get_cifar_100_datasets(None, None, split_train_val=False,
                         train_classes=range(80), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')