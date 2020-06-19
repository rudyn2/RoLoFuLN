import sys
import os
from dotenv import find_dotenv, load_dotenv
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from src.data.ds_handler import FashionMnistHandler
from src.models.simple_cnn import Network
from src.models.losses import DMILoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

load_dotenv(find_dotenv())
PROJECT_DIR = os.getenv('PROJECT_DIR')
sys.path.append(f"{PROJECT_DIR}")


class Solver:

    def __init__(self,
                 name: str,
                 model: nn.Module,
                 optimizer,
                 loss,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader):

        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.test_loader: DataLoader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.summary = SummaryWriter(f"{PROJECT_DIR}/data")

    def train(self, epochs: int, verbose: bool = False):
        """
        Generic training procedure.

        :param epochs:
        :param verbose:
        :return:
        """

        self.model.to(self.device)
        self.model.train()
        loader = tqdm(self.train_loader)

        for epoch in tqdm(range(epochs)):

            train_losses = []
            n_correct, total_items = 0.0, 0.0

            for i, data in enumerate(loader):
                inputs, labels = data
                inputs.to(self.device)
                labels.to(self.device)

                # Optimization step
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Getting the avg loss
                train_losses.append(loss.item())

                # calculating accuracy
                outputs = F.softmax(outputs, dim=1)
                batch_correct = (outputs.argmax(dim=1) == labels).float().sum()
                n_correct += batch_correct
                total_items += len(inputs)

                # Reporting actual batch results
                loader.set_description(
                    f'Epoch: {epoch + 1}  - Loss: {np.mean(train_losses[-20:]):.4f} ')

            # reports
            train_acc = 100 * n_correct / total_items
            val_loss, val_acc = self._get_validation_results()

            self.summary.add_scalar('Loss/train', np.mean(train_losses), epoch+1)
            self.summary.add_scalar('Loss/val', val_loss, epoch+1)
            self.summary.add_scalar('Accuracy/train', train_acc, epoch+1)
            self.summary.add_scalar('Accuracy/val', val_acc, epoch+1)

            if verbose and (epoch + 1) % 1 == 0:
                print(f"Epoch: {epoch + 1} || Train avg loss: {np.mean(train_losses):.4f} ||"
                      f"Train acc.: {train_acc:.3f} || Val avg. loss: {val_loss:.4f} "
                      f"|| Val avg. acc: {val_acc:.4f}")

    def _get_validation_results(self):
        """
        Returns the average loss, and accuracy of the input model evaluated
        in the data loader using the criterion.

        :return:                  Loss (float), Accuracy (float)
        """
        self.model.eval()
        val_loss, correct, total_items = 0, 0, 0

        with torch.no_grad():
            for inputs, target in self.val_loader:
                inputs.to(self.device)
                target.to(self.device)

                outputs = self.model(inputs)
                outputs = F.softmax(outputs, dim=1)

                val_loss += self.loss(outputs, target).item() / len(self.val_loader)
                correct += (outputs.argmax(dim=1) == target).float().sum()
                total_items += target.size(0)

        val_acc = 100. * correct / total_items
        return val_loss, val_acc


if __name__ == '__main__':
    data_dir = f'{PROJECT_DIR}/src/data/data'
    dataset = FashionMnistHandler(data_dir, False)
    dataset.load()
    train_loader, val_loader, test_loader = dataset.get_loaders(val_size=1/6)
    model = Network()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    solver = Solver('test', model, optimizer, criterion, train_loader, val_loader, test_loader)
    solver.train(epochs=5, verbose=True)
