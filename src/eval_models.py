from pathlib import Path
import os
import glob
from src.simple_cnn import CNNModel
from src.solvers import Solver, Summary
from src.ds_handler import FashionMnistHandler
import torch
import matplotlib.pyplot as plt


def get_noise_rate_from_name(filename: str):
    noise_rate = filename.split(".")[-2]
    return float(noise_rate) / 10


def eval_models(models_paths: list, path_to_data: str):

    test_acc = []
    for model_file in models_paths:

        # creating model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_file, map_location=device)
        model_name = model_file.split("/")[-1]

        # loading from checkpoint
        model = CNNModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()

        # getting test loader
        ds = FashionMnistHandler(path_to_data, False)
        ds.download()
        ds.load()
        # noise parameters are not relevant since test loader shouldn't have noise
        _, _, test_loader = ds.get_noisy_loaders(0, '1', 0.2, 64, 64, 64)

        # evaluating
        _, acc = Solver.eval(model, device, loss_fn=loss_fn, data_loader=test_loader)
        test_acc.append(acc)
        print(f"Model {model_name} has {acc:.4f} acc in test dataset")

    return test_acc


if __name__ == '__main__':
    parent_dir = str(Path(os.getcwd()).parent)

    data_dir = parent_dir + '/data'
    dmi_1_models_files = glob.glob(parent_dir + '/models/model_CNN_DMILoss_1_*.pt')
    dmi_1_models_files = sorted(dmi_1_models_files, key=get_noise_rate_from_name)
    ce_1_models_files = glob.glob(parent_dir + '/models/model_CNN_CrossEntropyLoss_1_*.pt')
    ce_1_models_files = sorted(ce_1_models_files, key=get_noise_rate_from_name)

    dmi_test_acc_1 = eval_models(dmi_1_models_files, data_dir)
    ce_test_acc_1 = eval_models(ce_1_models_files, data_dir)

    noise_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    fig, axs = plt.subplots(ncols=3, figsize=(16, 8), sharey='row')
    ylim = [50, 100]

    axs[0].plot(noise_values, dmi_test_acc_1, label='DMI')
    axs[0].plot(noise_values, ce_test_acc_1, '--', label='CE')
    axs[0].set_ylim(ylim)
    axs[0].legend()
    axs[0].set(xlabel='Noise amount ($r$)', ylabel='Test Accuracy (%)')

    axs[1].set_ylim(ylim)
    axs[1].set(xlabel='Noise amount ($r$)')

    axs[2].set_ylim(ylim)
    axs[2].set(xlabel='Noise amount ($r$)')
    plt.show()



