import os
import sys

import torch
from dotenv import find_dotenv, load_dotenv

from src.data.ds_handler import FashionMnistHandler
from src.models.losses import DMILoss
from src.models.simple_cnn import Network
from src.models.solvers import Solver

load_dotenv(find_dotenv())
PROJECT_DIR = os.getenv('PROJECT_DIR')
sys.path.append(f"{PROJECT_DIR}")

if __name__ == '__main__':

    # general parameters
    data_dir = f'{PROJECT_DIR}/src/data/data'
    noise_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    losses = [DMILoss(num_classes=2)] * 3
    EPOCHS = 20

    for noise_value in noise_values:

        # RUN Experiments
        for loss, tp_noise in zip(losses, ['1', '2', '3']):
            loss_name = loss.__class__.__name__
            tag = noise_value if loss_name == 'DMILoss' else ''
            name = f'CNN_{loss_name}_{tag}'

            print(f"Training {name} ...")

            # data preparation
            dataset = FashionMnistHandler(data_dir, False)
            dataset.load()
            train_loader, val_loader, test_loader = dataset.get_noisy_loaders(p_noise=noise_value,
                                                                              type_noise=tp_noise,
                                                                              val_size=1 / 6)

            # model, optimizer
            model = Network()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            # train
            solver = Solver(name, model, optimizer, loss, train_loader, val_loader, test_loader)
            solver.train(epochs=EPOCHS, verbose=True)

            print(f"Completed training...")
