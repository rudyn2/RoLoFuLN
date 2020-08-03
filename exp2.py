import torch
import os
from models import CNNModel
from solvers import Summary, Solver
from losses import DMILoss
from ds_handler import FashionMnistHandler

if __name__ == '__main__':
    # general parameters
    PROJECT_DIR = '.'
    MODEL_DIR = 'models_exp2'
    SUMMARIES_DIR = 'summaries_exp2'
    torch.random.manual_seed(42)
    data_dir = f'{PROJECT_DIR}/data'

    noise_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    lr = 1e-4

    for batch_size in [32, 64, 128, 192, 256]:
        print(f"BATCH SIZE: {batch_size}")
        # train
        batch_model_dir = MODEL_DIR + f'/batch_{batch_size}'
        if not os.path.exists(batch_model_dir):
            os.makedirs(batch_model_dir)

        batch_summaries_dir = SUMMARIES_DIR + f'/batch_{batch_size}'
        if not os.path.exists(batch_summaries_dir):
            os.makedirs(batch_summaries_dir)

        for tp_noise in ['1', '2', '3']:
            print("NOISE: " + tp_noise)
            for loss in [torch.nn.CrossEntropyLoss(), DMILoss(num_classes=2)]:
                loss_name = loss.__class__.__name__
                print(f"Loss: {loss_name}\n")
                for noise_value in noise_values:
                    # RUN Experiments

                    name = f'CNN_{loss_name}_{tp_noise}_{noise_value}'

                    print(f"Training {name} with noise of type {tp_noise} and probability {noise_value}...")

                    # data preparation
                    dataset = FashionMnistHandler(data_dir, False)
                    dataset.load()
                    train_loader, val_loader, test_loader = dataset.get_noisy_loaders(p_noise=noise_value,
                                                                                      type_noise=tp_noise,
                                                                                      val_size=1 / 6,
                                                                                      train_batch_size=batch_size,
                                                                                      val_batch_size=128,
                                                                                      test_batch_size=128)

                    # model, optimizer, summary
                    model = CNNModel()
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    summ = Summary(name, type_noise=tp_noise, noise_rate=noise_value)

                    solver = Solver(name, PROJECT_DIR, batch_model_dir, batch_summaries_dir, model,
                                    optimizer, loss, summ, train_loader, val_loader, test_loader)
                    solver.pretrain()
                    solver.train(loss)

                    print(f"Completed training...")
