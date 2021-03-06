import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time


class Summary:
    """
    Auxiliary class to record train and validation losses and accuracies.
    """

    def __init__(self, name: str, type_noise: str, noise_rate: float):
        self.name = name

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        self.type_noise = type_noise
        self.noise_rate = noise_rate

    def record_metrics(self, train_loss: float, train_acc: float, val_loss: float, val_acc: float):
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)

    def plot_loss(self, ax: plt.Axes, color):
        return self.plot_metric(ax, color, self.train_losses, self.val_losses, 'Loss')

    def plot_acc(self, ax: plt.Axes, color):
        return self.plot_metric(ax, color, self.train_accuracies, self.val_accuracies, 'Accuracy')

    def plot_metric(self, ax: plt.Axes, color: str, train: list, val: list, ylabel: str):
        epochs = range(1, len(train) + 1)
        ax.plot(epochs, train, label=f'{self.name}', color=color)
        ax.plot(epochs, val, color=color, linestyle='dashed')
        ax.legend()
        ax.set(xlabel='# Epochs', ylabel=ylabel, title=f'{ylabel} vs Epochs')
        return ax

    def save(self, path: str):
        torch.save(self, path + f'/{self.name}')

    @classmethod
    def load(cls, path_to_summ: str):
        with open(path_to_summ, 'rb') as summ_file:
            return torch.load(summ_file)


class Solver:

    def __init__(self,
                 name: str,
                 project_dir: str,
                 model_dir: str,
                 summaries_dir: str,
                 model: nn.Module,
                 optimizer,
                 loss: torch.nn,
                 summary: Summary,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader):

        self.name = name
        self.project_dir = project_dir

        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.test_loader: DataLoader = test_loader

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.summary = summary
        self.checkpoint_path = f"{project_dir}/{model_dir}"
        self.summary_dir = f"{project_dir}/{summaries_dir}"

    def pretrain(self):
        print("Pretraining using Cross Entropy Loss...")
        self.train_procedure(torch.nn.CrossEntropyLoss(), epochs=10, verbose=True)
        print("Pretraining ready!")

    def train(self, loss_fn):
        self.train_procedure(loss_fn, epochs=10, verbose=True, save_best_model=True)

    def train_procedure(self, loss_fn, epochs: int, verbose: bool = False, save_best_model: bool = True):
        """
        Generic training procedure.

        :param loss_fn:
        :param epochs:
        :param verbose:
        :param save_best_model:
        """

        self.model.to(self.device)
        self.model.train()

        checkpoint = False
        min_val_loss = np.infty

        for epoch in range(epochs):

            start = time.time()
            train_losses = []
            n_correct, total_items = 0.0, 0.0

            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # region: Optimization step
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss_ = loss_fn(outputs, labels)
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
            train_loss = float(np.mean(train_losses))
            val_loss, val_acc = self.eval(self.model, self.device, loss_fn, self.val_loader)

            # checkpoint
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(dict(epoch=epoch + 1,
                                model_state_dict=self.model.state_dict(),
                                optimizer_state_dict=self.optimizer.state_dict(),
                                train_loss=train_loss,
                                train_acc=train_acc,
                                valid_loss=val_loss,
                                valid_acc=val_acc), self.checkpoint_path + f'/model_{self.name}.pt')
                checkpoint = True

            self.summary.record_metrics(train_loss, train_acc, val_loss, val_acc)

            if verbose and (epoch + 1) % 1 == 0:
                loss_tag = loss_fn.__class__.__name__
                delta = time.time() - start
                print(f"Epoch: {epoch + 1} ({loss_tag})({delta:.1f} sec) Train avg loss: {train_loss:.4f} ||"
                      f"Train acc.: {train_acc:.3f} || Val avg. loss: {val_loss:.4f} "
                      f"|| Val avg. acc: {val_acc:.4f}")
                if save_best_model and checkpoint:
                    checkpoint = False
                    print(f"Model saved in epoch {epoch + 1} with val acc: {val_acc}.")

        self.summary.save(path=f"{self.project_dir}/{self.summary_dir}")

    @classmethod
    def eval(cls, model, device, loss_fn, data_loader):

        model.eval()
        correct, total_items = 0, 0

        losses = []
        with torch.no_grad():
            for inputs, target in data_loader:
                inputs = inputs.to(device)
                target = target.to(device)

                outputs = model(inputs)
                outputs = F.softmax(outputs, dim=1)

                losses.append(loss_fn(outputs, target).item())
                correct += (outputs.argmax(dim=1) == target).float().sum()
                total_items += len(inputs)

        acc = 100. * correct / total_items
        loss = np.mean(losses)
        return float(loss), float(acc)
