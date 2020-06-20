import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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
        self.model.apply(self.weights_init)

        best_valid_acc = 0
        checkpoint = False
        ce_epochs = int(epochs/2)
        ce_loss = torch.nn.CrossEntropyLoss()

        for epoch in range(ce_epochs):

            train_losses = []
            n_correct, total_items = 0.0, 0.0

            working_loss = ce_loss if epoch <= ce_epochs else self.loss

            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs.to(self.device)
                labels.to(self.device)

                # region: Optimization step
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss_ = working_loss(outputs, labels)
                loss_.backward()
                self.optimizer.step()
                # endregion

                # Getting the avg loss
                train_losses.append(loss_.item())

                # calculating accuracy
                outputs = F.softmax(outputs, dim=1)
                batch_correct = (outputs.argmax(dim=1) == labels).float().sum()
                n_correct += batch_correct
                total_items += len(inputs)

            # reports
            train_acc = 100 * n_correct / total_items
            train_loss = np.mean(train_losses)
            val_loss, val_acc = self._get_validation_results()

            # checkpoint
            if val_acc > best_valid_acc:
                best_valid_acc = val_acc
                torch.save(dict(epoch=epoch + 1,
                                model_state_dict=self.model.state_dict(),
                                optimizer_state_dict=self.optimizer.state_dict(),
                                train_loss=train_loss,
                                train_acc=train_acc,
                                valid_loss=val_loss,
                                valid_acc=val_acc), self.checkpoint_path + f'/model_{self.name}.pt')
                checkpoint = True

            # region: monitoring
            self.summary.add_scalar(f'{self.name}-Loss/train', train_loss, epoch + 1)
            self.summary.add_scalar(f'{self.name}-Loss/val', val_loss, epoch + 1)
            self.summary.add_scalar(f'{self.name}-Accuracy/train', train_acc, epoch + 1)
            self.summary.add_scalar(f'{self.name}-Accuracy/val', val_acc, epoch + 1)
            # endregion

            if verbose and (epoch + 1) % 1 == 0:
                loss_tag = working_loss.__class__.__name__
                print(f"Epoch: {epoch + 1} ({loss_tag})|| Train avg loss: {train_loss:.4f} ||"
                      f"Train acc.: {train_acc:.3f} || Val avg. loss: {val_loss:.4f} "
                      f"|| Val avg. acc: {val_acc:.4f}")
                if checkpoint:
                    checkpoint = False
                    print(f"Model saved in epoch {epoch + 1} with val acc: {val_acc}.")

    def _get_validation_results(self):
        """
        Returns the average loss, and accuracy of the input model evaluated
        in the data loader using the criterion.

        :return:                  Loss (float), Accuracy (float)
        """
        self.model.eval()
        val_loss, correct, total_items = 0, 0, 0

        val_losses = []
        with torch.no_grad():
            for inputs, target in self.val_loader:
                inputs.to(self.device)
                target.to(self.device)

                outputs = self.model(inputs)
                outputs = F.softmax(outputs, dim=1)

                val_losses.append(self.loss(outputs, target).item())
                correct += (outputs.argmax(dim=1) == target).float().sum()
                total_items += len(inputs)

        val_acc = 100. * correct / total_items
        val_loss = np.mean(val_losses)
        return val_loss, val_acc

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight.data)
