import torch
import os
from models import CNNModel
from solvers import Summary, Solver
from losses import DMILoss
from ds_handler import FashionMnistHandler

if __name__ == '__main__':
    # general parameters
    PROJECT_DIR = '.'
    MODEL_DIR = 'models_exp3'
    SUMMARIES_DIR = 'summaries_exp3'
    torch.random.manual_seed(42)
    data_dir = f'{PROJECT_DIR}/data'

    # parameters
    tp_noise = '1'
    noise_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    lr = 1e-4

    noise_model_dir = MODEL_DIR
    if not os.path.exists(noise_model_dir):
        os.makedirs(noise_model_dir)

    noise_summaries_dir = SUMMARIES_DIR
    if not os.path.exists(noise_summaries_dir):
        os.makedirs(noise_summaries_dir)

    for n_noise in [5, 10, 15, 20]:

        print(f"N_NOISE_ADD: {n_noise}")
        # data preparation
        dataset = FashionMnistHandler(data_dir, False, n_noise=n_noise)
        dataset.clean_processed()
        dataset.download()
        dataset.load()
        # since the data already has noise, it isn't necessary to add noise so p_noise=0
        train_loader, val_loader, test_loader = dataset.get_noisy_loaders(p_noise=0,
                                                                          type_noise=tp_noise,
                                                                          val_size=1 / 6,
                                                                          train_batch_size=128,
                                                                          val_batch_size=128,
                                                                          test_batch_size=128)

        print(f"Training adding {n_noise} new examples per clean example...")

        for loss in [torch.nn.CrossEntropyLoss(), DMILoss(num_classes=2)]:
            loss_name = loss.__class__.__name__
            print(f"Loss: {loss_name}\n")
            # RUN Experiments
            name = f'CNN_{loss_name}_{n_noise}'

            # model, optimizer, summary
            model = CNNModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            summ = Summary(name, type_noise=tp_noise, noise_rate=0)

            # train
            print(f"Train loader length: {len(train_loader)}")
            solver = Solver(name, PROJECT_DIR, noise_model_dir, noise_summaries_dir, model,
                            optimizer, loss, summ, train_loader, val_loader, test_loader)
            solver.pretrain()
            solver.train(loss)

            print(f"Completed training...")
