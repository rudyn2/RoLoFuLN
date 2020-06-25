import os
from pathlib import Path

import torch

from src.ds_handler import FashionMnistHandler
from src.losses import DMILoss
from src.simple_cnn import CNNModel
from src.solvers import Solver, Summary

if __name__ == '__main__':

    PROJECT_DIR = str(Path(os.getcwd()).parent)
    torch.random.manual_seed(42)

    # general parameters
    data_dir = f'{PROJECT_DIR}/data'

    # parameters
    # loss = DMILoss(num_classes=2)
    loss = DMILoss(num_classes=2)
    tp_noise = '1'
    noise_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    EPOCHS = 1
    lr = 1e-3

    for noise_value in noise_values:
        # RUN Experiments
        loss_name = loss.__class__.__name__
        name = f'CNN_{loss_name}_{tp_noise}_{noise_value}'

        print(f"Training {name} with noise of type {tp_noise} and probability {noise_value}...")

        # data preparation
        dataset = FashionMnistHandler(data_dir, False)
        dataset.load()
        train_loader, val_loader, test_loader = dataset.get_noisy_loaders(p_noise=noise_value,
                                                                          type_noise=tp_noise,
                                                                          val_size=1 / 6,
                                                                          train_batch_size=64,
                                                                          val_batch_size=64,
                                                                          test_batch_size=64)

        # model, optimizer, summary
        model = CNNModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        summ = Summary(name, type_noise=tp_noise, noise_rate=noise_value)

        # train
        solver = Solver(name, PROJECT_DIR, model, optimizer, loss, summ, train_loader, val_loader, test_loader)
        solver.pretrain()
        solver.train(loss)

        print(f"Completed training...")
