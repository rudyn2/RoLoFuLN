# -*- coding: utf-8 -*-
import logging
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from dotenv import find_dotenv, load_dotenv
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets

# region: CONFIG STUB
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

logger = logging.getLogger(__name__)
logger.info("Loading ENV variables")
load_dotenv(find_dotenv())

# loading env vars
PROJECT_DIR = os.getenv('PROJECT_DIR')
RANDOM_SEED = int(os.getenv('RANDOM_SEED'))
TRAIN_BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE'))

np.random.seed(RANDOM_SEED)


# endregion


def get_cifar_10(download: bool = False, val_size: int = 0.2):
    """
    Downloads the CIFAR10 Dataset and returns its train, val and test loaders. The data is already normalized.
    """

    logger.info('Getting CIFAR 10 Data')

    data_dir = f'{PROJECT_DIR}/src/data/data'
    try:
        train_loader, val_loader = get_train_valid_loader(data_dir, TRAIN_BATCH_SIZE, val_size, download, False)
        test_loader = get_test_loader(data_dir, TRAIN_BATCH_SIZE, download)

        logger.info("CIFAR 10 was successfully loaded!")
        return train_loader, val_loader, test_loader

    except RuntimeError:
        raise RuntimeError("Dataset not found. Run get_dataset.py to download the data.")


def get_train_valid_loader(data_dir: str,
                           batch_size: int,
                           valid_size: int,
                           download: bool = False,
                           augment: bool = False,
                           shuffle: bool = True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - download: True if dataset needs to be downloaded. Otherwise the data dir will be read.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=download, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=download, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=4)

    return train_loader, valid_loader


def get_test_loader(data_dir: str,
                    batch_size: int,
                    download: bool = True,
                    shuffle: bool = True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - download: True if dataset needs to be downloaded. Otherwise the data dir will be read.
    - shuffle: whether to shuffle the dataset after every epoch.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=download, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=4)

    return data_loader


if __name__ == '__main__':
    # run this script before
    get_cifar_10(download=True)
