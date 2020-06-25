# -*- coding: utf-8 -*-
import click

from src.ds_handler import *


@click.command()
@click.option('--name', help='Dataset to download [CIFAR10, FashionMnist].')
@click.option('--path', help='Output path of data.')
@click.option('--augment', default=True, help='If true applies data augmentation to train dataset')
def download(name, path, augment):
    """Simple program that downloads datasets to local."""
    available_datasets = ["CIFAR10", "FashionMnist"]
    if name not in available_datasets:
        raise ValueError(f"Just the following datasets are currently availables to download: {available_datasets}")

    if name == "FashionMnist":
        FashionMnistHandler(path, augment).download()
    elif name == "CIFAR10":
        Cifar10Handler(path, augment).download()


if __name__ == '__main__':
    download()
