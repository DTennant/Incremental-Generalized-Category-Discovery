import os
import pandas as pd
import numpy as np
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from data.data_utils import subsample_instances
from config import inat21_root


class iNat(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader, download=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):

        self.data = pd.read_csv(os.path.join(self.root, 'train_val_mini.csv'), 
                              names=['img_id', 'filepath', 'target', 'cls_name', 'is_training_img'])

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        # for index, row in self.data.iterrows():
        #     filepath = os.path.join(self.root, row.filepath)
        #     if not os.path.isfile(filepath):
        #         print(filepath)
        #         return False

        return True


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, sample.filepath)
        target = int(sample.target)
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            try:
                target = self.target_transform(target)
            except KeyError:
                # import ipdb; ipdb.set_trace()
                print('target', target)
                print('idx', idx)
                raise ValueError

        return img, target, self.uq_idxs[idx]

class iNat_incremental(Dataset):

    def __init__(self, root, use_small=True, split_crit='location', stage=1, 
                return_path=False, train=True, transform=None, target_transform=None, 
                loader=default_loader, download=False):
        # stage indicates the stage of incremental learning
        # split_crit indicates the split criterion: can be location, year, or loc_year
        # use_small indicates whether to use the small dataset

        self.root = os.path.expanduser(root)
        self.stage = stage
        self.use_small = use_small
        self.split_crit = split_crit
        self.return_path = return_path
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):

        # self.data = pd.read_csv(os.path.join(self.root, 'train_val_mini.csv'), )
                              # names=['img_id', 'filepath', 'target', 'cls_name', 'is_training_img'])

        if self.train:
            self.data = pd.read_csv(os.path.join(self.root, 'incremental_splits', self.split_crit + '_small' if self.use_small else self.split_crit, f'stage_{self.stage}.csv'))
            # self.data = self.data[['image_id', 'category_id', 'file_name', 'date', 'continent', 'common_name', 'genus']]
        else:
            self.data = pd.read_csv(os.path.join(self.root, 'incremental_splits', self.split_crit + '_small', f'val_stage_{self.stage}.csv'))
            # self.data = self.data[['image_id', 'category_id', 'file_name', 'date', 'continent', 'common_name', 'genus']]
            
        self.data = self.data[['file_name', 'category_id']]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, sample.file_name)
        target = int(sample.category_id)
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            try:
                target = self.target_transform(target)
            except KeyError:
                # import ipdb; ipdb.set_trace()
                print('target', target)
                print('idx', idx)
                raise ValueError

        if self.return_path:
            return img, target, self.uq_idxs[idx], sample.file_name
        return img, target, self.uq_idxs[idx]


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes)
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.data['target'])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.data['target'] == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_inat21_datasets(train_transform, test_transform, train_classes=range(160), prop_train_labels=0.8,
                    split_train_val=False, seed=0, args=None):
    # import ipdb; ipdb.set_trace()
    np.random.seed(seed)

    # Init entire training set
    whole_training_set = iNat(root=inat21_root, transform=train_transform, train=True, )

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
    test_dataset = iNat(root=inat21_root, transform=test_transform, train=False)

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

def get_inat21_incremental_datasets(train_transform, test_transform, train_classes=range(160), prop_train_labels=0.8,
                    split_train_val=False, seed=0, args=None):
    # import ipdb; ipdb.set_trace()
    # train_classes, prop_train_labels, split_train_val, seed will be ignored
    np.random.seed(seed)

    # Init entire training set
    whole_training_set = iNat_incremental(root=inat21_root, use_small=args.use_small_set, split_crit=args.split_crit, stage=args.stage, return_path=args.return_path, transform=train_transform, train=True, )

    next_stage = args.stage + 1 if args.stage < args.max_stage else args.stage
    train_unlabelled = iNat_incremental(root=inat21_root, use_small=args.use_small_set, split_crit=args.split_crit, stage=next_stage, 
                                    transform=train_transform, train=True, return_path=args.return_path)
    test_dataset = iNat_incremental(root=inat21_root, use_small=args.use_small_set, split_crit=args.split_crit, stage=next_stage, return_path=args.return_path, transform=test_transform, train=False)

    args.root = inat21_root
    args.classes_in_labelled = set(whole_training_set.data.category_id)
    args.classes_in_unlabelled = set(train_unlabelled.data.category_id) - set(args.classes_in_labelled)
    args.classes_in_labelled = args.classes_in_labelled - args.classes_in_unlabelled
    
    args.train_classes = args.classes_in_labelled
    args.unlabeled_classes = args.classes_in_unlabelled

    all_datasets = {
        'train_labelled': whole_training_set,
        'train_unlabelled': train_unlabelled,
        'val': None,
        'test': test_dataset,
    }

    return all_datasets


if __name__ == '__main__':

    x = get_cub_datasets(None, None, split_train_val=False,
                         train_classes=range(100), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].data["target"].values))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].data["target"].values))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')
    
