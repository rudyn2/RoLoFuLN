import torch
import os
from models import CNNModel
from solvers import Summary, Solver
from losses import DMILoss
from ds_handler import FashionMnistHandler

if __name__ == '__main__':
    PROJECT_DIR = '.'
    MODEL_DIR = 'models_exp1'
    SUMMARIES_DIR = 'summaries_exp1'
    torch.random.manual_seed(42)

    # general parameters
    data_dir = f'{PROJECT_DIR}/data'

    # parameters
    noise_values = [0.7, 0.8, 0.9]
    lr = 1e-4

    for tp_noise in ['1']:
        print("NOISE: " + tp_noise)

        tp_noise_model_dir = MODEL_DIR + f'/type_{tp_noise}'
        if not os.path.exists(tp_noise_model_dir):
            os.makedirs(tp_noise_model_dir)
        tp_noise_summaries_dir = SUMMARIES_DIR + f'/type_{tp_noise}'

        if not os.path.exists(tp_noise_summaries_dir):
            os.makedirs(tp_noise_summaries_dir)

        for noise_value in noise_values:

            # data preparation
            dataset = FashionMnistHandler(data_dir, False)
            dataset.load()
            train_loader, val_loader, test_loader = dataset.get_noisy_loaders(p_noise=noise_value,
                                                                              type_noise=tp_noise,
                                                                              val_size=1 / 6,
                                                                              train_batch_size=128,
                                                                              val_batch_size=128,
                                                                              test_batch_size=128)

            # RUN Experiments
            for loss in [torch.nn.CrossEntropyLoss(), DMILoss(num_classes=2)]:
                loss_name = loss.__class__.__name__
                name = f'CNN_{loss_name}_{tp_noise}_{noise_value}'
                print(f"Training {name} with noise of type {tp_noise} and probability {noise_value}...")

                # model, optimizer, summary
                model = CNNModel()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                summ = Summary(name, type_noise=tp_noise, noise_rate=noise_value)

                # train
                solver = Solver(name, PROJECT_DIR, tp_noise_model_dir, tp_noise_summaries_dir, model,
                                optimizer, loss, summ, train_loader, val_loader, test_loader)
                solver.pretrain()
                solver.train(loss)

                print(f"Completed training...")
