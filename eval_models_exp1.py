import glob
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
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
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
    model_dir = parent_dir + '/models_exp1'
    plot_noise_1 = True
    plot_noise_2 = True
    plot_noise_3 = True

    fig, axs = plt.subplots(ncols=3, figsize=(16, 5), sharey='row')
    noise_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if plot_noise_1:
        dmi_models_files_1 = glob.glob(parent_dir + f'/models_exp1/type_1/model_CNN_DMILoss_1_*.pt')
        dmi_models_files_1 = sorted(dmi_models_files_1, key=get_noise_rate_from_name)
        ce_models_files_1 = glob.glob(parent_dir + f'/models_exp1/type_1/model_CNN_CrossEntropyLoss_1_*.pt')
        ce_models_files_1 = sorted(ce_models_files_1, key=get_noise_rate_from_name)
        dmi_test_acc_1 = eval_models(dmi_models_files_1, data_dir)
        ce_test_acc_1 = eval_models(ce_models_files_1, data_dir)

        axs[0].plot(noise_values, dmi_test_acc_1, '^-', linewidth=3, label='DMI')
        axs[0].plot(noise_values, ce_test_acc_1, '*--', linewidth=3, label='CE')
        axs[0].set(xlabel='Noise amount ($r$)', ylabel='Test Accuracy (%)')
        axs[0].legend()

    if plot_noise_2:
        dmi_models_files_2 = glob.glob(parent_dir + f'/models_exp1/type_2/model_CNN_DMILoss_2_*.pt')
        dmi_models_files_2 = sorted(dmi_models_files_2, key=get_noise_rate_from_name)
        ce_models_files_2 = glob.glob(parent_dir + f'/models_exp1/type_2/model_CNN_CrossEntropyLoss_2_*.pt')
        ce_models_files_2 = sorted(ce_models_files_2, key=get_noise_rate_from_name)
        dmi_test_acc_2 = eval_models(dmi_models_files_2, data_dir)
        ce_test_acc_2 = eval_models(ce_models_files_2, data_dir)

        axs[1].plot(noise_values, dmi_test_acc_2, '^-', linewidth=3, label='DMI')
        axs[1].plot(noise_values, ce_test_acc_2, '*--', linewidth=3, label='CE')
        axs[1].set(xlabel='Noise amount ($r$)')

    if plot_noise_3:

        dmi_models_files_3 = glob.glob(parent_dir + f'/models_exp1/type_3/model_CNN_DMILoss_3_*.pt')
        dmi_models_files_3 = sorted(dmi_models_files_3, key=get_noise_rate_from_name)
        ce_models_files_3 = glob.glob(parent_dir + f'/models_exp1/type_3/model_CNN_CrossEntropyLoss_3_*.pt')
        ce_models_files_3 = sorted(ce_models_files_3, key=get_noise_rate_from_name)
        dmi_test_acc_3 = eval_models(dmi_models_files_3, data_dir)
        ce_test_acc_3 = eval_models(ce_models_files_3, data_dir)

        axs[2].plot(noise_values, dmi_test_acc_3, '^-', linewidth=3, label='DMI')
        axs[2].plot(noise_values, ce_test_acc_3, '*--', linewidth=3, label='CE')
        axs[2].set(xlabel='Noise amount ($r$)')

    plt.show()
