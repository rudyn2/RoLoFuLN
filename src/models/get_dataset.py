# -*- coding: utf-8 -*-

import os
import sys

import click
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
PROJECT_DIR = os.getenv('PROJECT_DIR')
sys.path.append(f"{PROJECT_DIR}")

from src.models.ds_handler import Cifar10Handler, FashionMnistHandler


@click.command()
@click.option('--name', help='Dataset to download [CIFAR10, FashionMnist].')
@click.option('--augment', default=True, help='If true applies data augmentation to train dataset')
def download(name, augment):
    """Simple program that downloads datasets to local."""
    available_datasets = ["CIFAR10", "FashionMnist"]
    if name not in available_datasets:
        raise ValueError(f"Just the following datasets are currently availables to download: {available_datasets}")

    data_dir = f'{PROJECT_DIR}/src/data/data'
    if name == "FashionMnist":
        FashionMnistHandler(data_dir, augment).download()
    elif name == "CIFAR10":
        Cifar10Handler(data_dir, augment).download()


if __name__ == '__main__':
    download()
