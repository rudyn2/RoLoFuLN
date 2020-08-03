import torch
import os
from models import CNNModel, MLP
from solvers import Summary, Solver
from losses import DMILoss
from ds_handler import FashionMnistHandler

if __name__ == '__main__':
    PROJECT_DIR = '.'
    MODEL_DIR = 'models_exp4'
    SUMMARIES_DIR = 'summaries_exp4'
    torch.random.manual_seed(42)

    # general parameters
    data_dir = f'{PROJECT_DIR}/data'

    # parameters
    lr = 1e-4

    mlp1 = MLP(input_dim=28*28, hidden_dim=25, hidden_layers=1, output_dim=2)
    mlp2 = MLP(input_dim=28 * 28, hidden_dim=25, hidden_layers=2, output_dim=2)
    mlp4 = MLP(input_dim=28 * 28, hidden_dim=25, hidden_layers=4, output_dim=2)
    loss = DMILoss(num_classes=2)
    noise_values = [0.5, 0.7, 0.9]

    for model, n_layers in zip([mlp1], [1]):
        model_name = model.__class__.__name__ + str(n_layers)
        print(f"MODEL: {model_name}")

        for tp_noise in ['3']:

            # region: prepare dir
            tp_noise_model_dir = MODEL_DIR + f'/type_{tp_noise}'
            if not os.path.exists(tp_noise_model_dir):
                os.makedirs(tp_noise_model_dir)
            tp_noise_summaries_dir = SUMMARIES_DIR + f'/type_{tp_noise}'
            # endregion

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
                loss_name = loss.__class__.__name__
                name = f'{model_name}_{loss_name}_{tp_noise}_{noise_value}'
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