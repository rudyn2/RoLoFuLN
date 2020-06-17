import sys
import os
from dotenv import find_dotenv, load_dotenv
import torchvision.models as models
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from src.data.ds_handler import DatasetHandler, FashionMnistHandler

load_dotenv(find_dotenv())
PROJECT_DIR = os.getenv('PROJECT_DIR')
sys.path.append(f"{PROJECT_DIR}")


class Solver:

    def __init__(self,
                 name: str,
                 model: nn.Module,
                 optimizer,
                 loss,
                 dataset_handler: DatasetHandler,
                 gpu: bool = False):

        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.dataset_handler: DatasetHandler = dataset_handler
        self.gpu = gpu
        self.summary = SummaryWriter(f"{PROJECT_DIR}/data")

    def train(self, epochs: int, verbose: bool = False):

        if self.gpu:
            self.model.cuda()

        self.model.train()
        train_loader, val_loader, _ = self.dataset_handler.get_loaders(batch_size=32,
                                                                       val_size=0.2,
                                                                       shuffle=True)
        for epoch in range(epochs):

            train_loss, n_correct, total_items = 0.0, 0.0, 0.0

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = torch.flatten(inputs, start_dim=1)

                if self.gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                loss, outputs = self._optimizer_step(inputs, labels)

                # Getting the avg loss
                train_loss += loss.item() / len(train_loader)

                # calculating accuracy
                batch_correct = (outputs.argmax(dim=1) == labels).float().sum()
                n_correct += batch_correct
                total_items += len(inputs)

            # reports
            train_acc = 100 * n_correct / total_items
            val_loss, val_acc = self._get_validation_results(val_loader)

            self.summary.add_scalars(f'run_{self.name}',
                                     {
                                         'accuracy/train': train_acc,
                                         'accuracy/val': val_acc,
                                         'loss/train': train_loss,
                                         'loss/val': val_loss
                                     })

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch: {epoch + 1} || Train avg loss: {train_loss:.4f} ||"
                      f"Train acc.: {train_acc:.3f} || Val avg. loss: {val_loss:.4f} "
                      f"|| Val avg. acc: {val_acc:.4f}")

    def _optimizer_step(self, inputs, labels):

        # Optimization step
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss, outputs

    def _get_validation_results(self, data_loader):
        """
        Returns the average loss, and accuracy of the input model evaluated
        in the data loader using the criterion.

        :return:                  Loss (float), Accuracy (float)
        """
        self.model.eval()
        val_loss, correct = 0, 0

        with torch.no_grad():
            for data, target in data_loader:
                data = torch.flatten(data, start_dim=1)

                if self.gpu:
                    data = data.cuda()
                    target = target.cuda()

                outputs = self.model(data)
                val_loss += self.loss(outputs, target).item() / len(data_loader)
                correct += (outputs.argmax(dim=1) == target).float().sum()

        val_acc = 100. * correct / len(data_loader.dataset)
        return val_loss, val_acc


if __name__ == '__main__':
    data_dir = f'{PROJECT_DIR}/src/data/data'
    dataset = FashionMnistHandler(data_dir, False)
    dataset.load()
    resnet = models.resnet34(pretrained=False, progress=False)
    optimizer = torch.optim.Adam(resnet.parameters())
    criterion = nn.CrossEntropyLoss()
    solver = Solver('test', resnet, optimizer, criterion, dataset)
    solver.train(2, True)
