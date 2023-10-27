from data.data_utils import MergedDataset

from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets, get_cifar_100_lt_datasets
from data.herbarium_19 import get_herbarium_datasets
from data.stanford_cars import get_scars_datasets
from data.imagenet import get_imagenet_100_datasets, get_imagenet_100_lt_datasets
from data.cub import get_cub_datasets
from data.fgvc_aircraft import get_aircraft_datasets
from data.nabirds import get_nabirds_datasets
from data.inat21 import get_inat21_datasets, get_inat21_incremental_datasets

from copy import deepcopy
import pickle
import os
import numpy as np

from config import osr_split_dir


get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'cifar100_lt': get_cifar_100_lt_datasets,
    'cifar100_osssl': get_cifar_100_datasets,
    'imagenet_100': get_imagenet_100_datasets,
    'imagenet_100_lt': get_imagenet_100_lt_datasets,
    'herbarium_19': get_herbarium_datasets,
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets,
    'nabirds': get_nabirds_datasets,
    'inat21': get_inat21_datasets,
    'inat21_incremental': get_inat21_incremental_datasets,
}


def get_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    # import ipdb; ipdb.set_trace()
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                            train_classes=args.train_classes,
                            prop_train_labels=args.prop_train_labels,
                            split_train_val=False,
                            args=args)
    # Set target transforms:
    if args.existing_mapping is not None:
        for c in list(args.train_classes) + list(args.unlabeled_classes):
            if c not in args.existing_mapping.keys():
                args.existing_mapping[c] = len(args.existing_mapping)
        target_transform_dict = args.existing_mapping
    else:
        target_transform_dict = {}
        for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
            target_transform_dict[cls] = i
        args.existing_mapping = target_transform_dict

    def _transform(x):
        return target_transform_dict[x]
            
    target_transform = _transform

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform

    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets


def get_class_splits(args):

    # For FGVC datasets, optionally return bespoke splits
    if args.dataset_name in ('scars', 'cub', 'aircraft', 'nabirds'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10':

        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset_name == 'cifar100':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'cifar100_osssl':

        args.image_size = 32
        args.train_classes = range(55)
        args.unlabeled_classes = range(55, 100)

    elif args.dataset_name == 'cifar100_lt':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']

    elif args.dataset_name == 'imagenet_100':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'imagenet_100_lt':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'scars':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']
            args.hard_classes = open_set_classes['Hard']
            args.medium_classes = open_set_classes['Medium']
            args.easy_classes = open_set_classes['Easy']

        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

    elif args.dataset_name == 'aircraft':

        args.image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']
            args.hard_classes = open_set_classes['Hard']
            args.medium_classes = open_set_classes['Medium']
            args.easy_classes = open_set_classes['Easy']

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cub':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']
            args.hard_classes = open_set_classes['Hard']
            args.medium_classes = open_set_classes['Medium']
            args.easy_classes = open_set_classes['Easy']

        else:

            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)
            
    elif args.dataset_name == 'nabirds':
        
        args.image_size = 224
        if use_ssb_splits:
            split_path = os.path.join(osr_split_dir, 'nabirds_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)
                
            args.train_classes = class_info['Label']['id']
            args.unlabeled_classes = class_info['Hard']['id'] + class_info['Medium']['id'] + class_info['Easy']['id']

            args.hard_classes = class_info['Hard']['id']
            args.medium_classes = class_info['Medium']['id']
            args.easy_classes = class_info['Easy']['id']

        else:
            raise ValueError('Only SSB splits for NABirds')
            args.train_classes = range(200)
            args.unlabeled_classes = range(200, 555)

    elif args.dataset_name == 'inat21': 
            
        args.image_size = 224
        all_cls = np.arange(0, 10000)
        np.random.seed(42)
        args.train_classes = np.random.choice(all_cls, size=len(all_cls)//2, replace=False).tolist()
        args.unlabeled_classes = list(set(all_cls) - set(args.train_classes))

    elif args.dataset_name == 'inat21_incremental': 
            
        args.image_size = 224
        all_cls = np.arange(0, 10000)
        np.random.seed(42)
        args.train_classes = np.random.choice(all_cls, size=len(all_cls)//2, replace=False).tolist()
        args.unlabeled_classes = list(set(all_cls) - set(args.train_classes))
        
    elif args.dataset_name == 'inat18':
        args.image_size = 224
        all_cls = np.arange(0, 8142)
        np.random.seed(42)
        args.train_classes = np.random.choice(all_cls, size=len(all_cls)//2, replace=False).tolist()
        args.unlabeled_classes = list(set(all_cls) - set(args.train_classes))

    elif args.dataset_name == 'inat19':
        #TODO: SSB splits
        args.image_size = 224
        all_cls = np.arange(0, 1010)
        np.random.seed(42)
        args.train_classes = np.random.choice(all_cls, size=len(all_cls)//2, replace=False).tolist()
        args.unlabeled_classes = list(set(all_cls) - set(args.train_classes))


    else:

        raise NotImplementedError

    if args.n_overlap_cls != -1:
        # move args.n_overlap_cls classes from train to unlabeled
        np.random.seed(42)
        args.overlap_cls = np.random.choice(args.train_classes, size=args.n_overlap_cls, replace=False).tolist()
    else:
        args.overlap_cls = []

    return args
