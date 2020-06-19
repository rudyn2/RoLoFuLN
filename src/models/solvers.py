import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
        self.summary = SummaryWriter(f"{PROJECT_DIR}/data/summaries")
        self.checkpoint_path = f"{PROJECT_DIR}/models"

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

        best_valid_loss, _ = self._get_validation_results()

        for epoch in range(epochs):

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
            train_loss = np.mean(train_losses)
            val_loss, val_acc = self._get_validation_results()

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                torch.save(dict(epoch=epoch + 1,
                                model_state_dict=self.model.state_dict(),
                                optimizer_state_dict=self.optimizer.state_dict(),
                                train_loss=train_loss,
                                valid_loss=val_loss), self.checkpoint_path + f'/model_{self.name}.pt')

            self.summary.add_scalar('Loss/train', train_loss, epoch + 1)
            self.summary.add_scalar('Loss/val', val_loss, epoch + 1)
            self.summary.add_scalar('Accuracy/train', train_acc, epoch + 1)
            self.summary.add_scalar('Accuracy/val', val_acc, epoch + 1)

            if verbose and (epoch + 1) % 1 == 0:
                print(f"Epoch: {epoch + 1} || Train avg loss: {train_loss:.4f} ||"
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
