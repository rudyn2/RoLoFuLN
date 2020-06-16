# -*- coding: utf-8 -*-
import os
from abc import abstractmethod

import numpy as np
import torch
import torchvision.transforms as transforms
from dotenv import find_dotenv, load_dotenv
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets


load_dotenv(find_dotenv())

# loading env vars
RANDOM_SEED = int(os.getenv('RANDOM_SEED'))
TRAIN_BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE'))
VAL_BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE'))
TEST_BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE'))

np.random.seed(RANDOM_SEED)


class DatasetHandler:
    """
    Auxiliary class used to handle and store pytorch datasets.
    """

    def __init__(self,
                 dataset,
                 data_dir: str,
                 augment: bool):
        """
        Constructor for DatasetHandler.

        :param dataset: Pytorch Dataset.
        :param data_dir: path directory to the dataset.
        :param augment: whether to apply the data augmentation scheme. Only applied on the train split.
        """
        self.dataset = dataset

        if not os.path.exists(data_dir):
            raise ValueError("Data path doesn't exist. Please enter a valid path.")

        self.data_dir = data_dir
        self.augment = augment

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    @property
    @abstractmethod
    def train_transform(self) -> transforms.Compose:
        raise NotImplementedError

    @property
    @abstractmethod
    def valid_transform(self) -> transforms.Compose:
        raise NotImplementedError

    @property
    @abstractmethod
    def test_transform(self) -> transforms.Compose:
        raise NotImplementedError

    def download(self):
        # downloads train set
        self.dataset(root=self.data_dir,
                     train=True,
                     download=True,
                     transform=self.train_transform)
        # downloads test set
        self.dataset(root=self.data_dir,
                     train=False,
                     download=True,
                     transform=self.test_transform)

    def load(self):
        # load the dataset
        self.train_dataset = self.dataset(root=self.data_dir,
                                          train=True,
                                          download=False,
                                          transform=self.train_transform)
        self.valid_dataset = self.dataset(root=self.data_dir,
                                          train=True,
                                          download=False,
                                          transform=self.valid_transform)
        self.test_dataset = self.dataset(root=self.data_dir,
                                         train=False,
                                         download=False,
                                         transform=self.test_transform)

    def _split(self, valid_size: float, shuffle: bool):

        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

        num_train = len(self.train_dataset)
        idxs = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(idxs)

        train_idx, valid_idx = idxs[split:], idxs[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        return train_sampler, valid_sampler

    def get_loaders(self, batch_size: int, val_size: float = 0, shuffle: bool = False):

        train_sampler, valid_sampler = self._split(val_size, shuffle)

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=TRAIN_BATCH_SIZE, sampler=train_sampler,
            num_workers=4)
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset, batch_size=VAL_BATCH_SIZE, sampler=valid_sampler,
            num_workers=4)
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=TEST_BATCH_SIZE, num_workers=4)

        return train_loader, valid_loader, test_loader


class Cifar10Handler(DatasetHandler):

    def __init__(self, data_dir: str, augment: bool = True):
        super(Cifar10Handler, self).__init__(datasets.CIFAR10, data_dir, augment)

        self.train_normalizer = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )

        self.test_normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    @property
    def train_transform(self) -> transforms.Compose:

        if self.augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.train_normalizer,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                self.train_normalizer,
            ])
        return train_transform

    @property
    def valid_transform(self):

        return transforms.Compose([
            transforms.ToTensor(),
            self.train_normalizer,
        ])

    @property
    def test_transform(self) -> transforms.Compose:

        # define transform
        return transforms.Compose([
            transforms.ToTensor(),
            self.test_normalizer,
        ])


class FashionMnistHandler(DatasetHandler):

    def __init__(self, data_dir: str, augment: bool = True):
        super(FashionMnistHandler, self).__init__(datasets.FashionMNIST, data_dir, augment)

    @property
    def train_transform(self) -> transforms.Compose:
        return transforms.Compose([])

    @property
    def valid_transform(self):
        return transforms.Compose([])

    @property
    def test_transform(self) -> transforms.Compose:
        # define transform
        return transforms.Compose([])

