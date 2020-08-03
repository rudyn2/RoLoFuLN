import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ds_handler import FashionMnistHandler
from models import CNNModel
from solvers import Solver

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE - 6)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_noise_rate_from_name(filename: str):
    noise_rate = filename.split(".")[-2]
    return float(noise_rate) / 10


def eval_models(models_paths: list, path_to_data: str):
    if len(models_paths) == 0:
        return 0.0

    # getting test loader
    ds = FashionMnistHandler(path_to_data, False)
    ds.download()
    ds.load()
    # noise parameters are not relevant since test loader shouldn't have noise
    _, _, test_loader = ds.get_noisy_loaders(0, '1', 0.2, 128, 128, 128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_acc = []
    for model_file in models_paths:
        # creating model
        checkpoint = torch.load(model_file, map_location=device)
        model_name = model_file.split("/")[-1]

        # loading from checkpoint
        model = CNNModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()

        # evaluating
        _, acc = Solver.eval(model, device, loss_fn=loss_fn, data_loader=test_loader)
        test_acc.append(acc)
        print(f"Model {model_name} has {acc:.4f} acc in test dataset")

    return test_acc


if __name__ == '__main__':
    parent_dir = '.'
    data_dir = parent_dir + '/data'
    model_dir = parent_dir + '/models_exp4'

    fig, axs = plt.subplots(ncols=3, figsize=(16, 5))
    noise_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for idx, tp_noise in enumerate(['1', '2', '3']):

        # CNN
        cnn_models_paths = glob.glob(parent_dir + '/models_exp1' + f'/type_{tp_noise}/model_CNN_DMILoss_{tp_noise}_*.pt')
        cnn_models_paths = sorted(cnn_models_paths, key=get_noise_rate_from_name)
        cnn_models_paths = [path for path in cnn_models_paths if get_noise_rate_from_name(path) in [0.1, 0.3, 0.5, 0.7, 0.9]]
        cnn_models_acc = eval_models(cnn_models_paths, data_dir)

        # MLP1
        mlp1_models_paths = glob.glob(model_dir + f'/type_{tp_noise}/model_MLP1_DMILoss_{tp_noise}_*.pt')
        mlp1_models_paths = sorted(mlp1_models_paths, key=get_noise_rate_from_name)
        mlp1_models_acc = eval_models(mlp1_models_paths, data_dir)

        # MLP2
        mlp2_models_paths = glob.glob(model_dir + f'/type_{tp_noise}/model_MLP2_DMILoss_{tp_noise}_*.pt')
        mlp2_models_paths = sorted(mlp2_models_paths, key=get_noise_rate_from_name)
        mlp2_models_acc = eval_models(mlp2_models_paths, data_dir)

        # MLP4
        mlp4_models_paths = glob.glob(model_dir + f'/type_{tp_noise}/model_MLP4_DMILoss_{tp_noise}_*.pt')
        mlp4_models_paths = sorted(mlp4_models_paths, key=get_noise_rate_from_name)
        mlp4_models_acc = eval_models(mlp4_models_paths, data_dir)

        axs[idx].plot(noise_values, cnn_models_acc, '^-', linewidth=3, label='CNN')
        axs[idx].plot(noise_values, mlp1_models_acc, '*--', linewidth=3, label='MLP1')
        axs[idx].plot(noise_values, mlp2_models_acc, '^-', linewidth=3, label='MLP2')
        axs[idx].plot(noise_values, mlp4_models_acc, '*--', linewidth=3, label='MLP4')
        axs[idx].set(xlabel='Noise amount ($r$)', ylabel='Test Accuracy (%)')
        axs[idx].legend()

    plt.show()
