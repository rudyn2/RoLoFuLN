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
    model_dir = parent_dir + '/models_exp3'

    fig, ax = plt.subplots(figsize=(8, 6))
    noise_values = [0, 5, 10, 15, 20]

    dmi_test_acc = []
    ce_test_acc = []
    for n_noise in noise_values:
        base_dir = parent_dir + f'/models_exp3'
        dmi_model_path = base_dir + f'/model_CNN_DMILoss_{n_noise}.pt'
        ce_model_path = base_dir + f'/model_CNN_CrossEntropyLoss_{n_noise}.pt'
        dmi_test_acc.append(eval_models([dmi_model_path], data_dir)[0])
        ce_test_acc.append(eval_models([ce_model_path], data_dir)[0])

    ax.plot(noise_values, dmi_test_acc, '^-', linewidth=3, label='LDMI')
    ax.plot(noise_values, ce_test_acc, '*--', linewidth=3, label='CE')
    ax.set(xlabel='Number of noisy labels per clean label', ylabel='Test Accuracy (%)')
    ax.legend()

    plt.show()
