import sys
import os
from dotenv import find_dotenv, load_dotenv
import torchvision.models as models
import torch

load_dotenv(find_dotenv())
PROJECT_DIR = os.getenv('PROJECT_DIR')
sys.path.append(f"{PROJECT_DIR}")


def get_validation_results(model, data_loader, criterion):
    """
    Returns the average loss, and accuracy of the input model evaluated
    in the data loader using the criterion.

    :param model:             PyTorch Model (nn.Module)
    :param data_loader:       PyTorch DataLoader
    :param criterion:         PyTorch Loss (torch.nn.loss)
    :return:                  Loss (float), Accuracy (float)
    """
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = torch.flatten(data, start_dim=1)
            data = data.cuda()
            target = target.cuda()
            outputs = model(data)
            val_loss += criterion(outputs, target).item() / len(data_loader)
            correct += (outputs.argmax(dim=1) == target).float().sum()

    val_acc = 100. * correct / len(data_loader.dataset)
    return val_loss, val_acc


def train_mlp(model, optimizer, criterion, lr_scheduler,
              train_loader, val_loader, summary,
              epochs=8, verbose=True):
    """
    Trains a simple mlp classifier model using some optimizer
    method and a pre defined criterion.

    :param model:         PyTorch Model (nn.Module)
    :param optimizer:     PyTorch optimizer
    :param criterion:     PyTorch Criterion (nn.Loss)
    :param summary:       Summary object.
    :param epochs:        Number of epochs to train (integer).
    :param verbose:       If true, the results will be displayed in the prompt (boolean)
    """

    model.train()

    for epoch in range(epochs):

        train_loss = 0.0
        n_correct = 0
        total_items = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = torch.flatten(inputs, start_dim=1)
            inputs = inputs.cuda()
            labels = labels.cuda()

            # Optimization step
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Getting the avg loss
            train_loss += loss.item() / len(train_loader)

            # calculating accuracy

            batch_correct = (outputs.argmax(dim=1) == labels).float().sum()
            n_correct += batch_correct
            total_items += len(inputs)

        # reports
        train_acc = 100 * n_correct / total_items
        val_loss, val_acc = get_validation_results(model, val_loader, criterion)

        summary.record_metrics(train_loss, train_acc, val_loss, val_acc)

        lr_scheduler.step()

        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch: {epoch + 1} || Train avg loss: {train_loss:.4f} ||"
                  f"Train acc.: {train_acc:.3f} || Val avg. loss: {val_loss:.4f} "
                  f"|| Val avg. acc: {val_acc:.4f}")
    return summary


if __name__ == '__main__':
    resnet = models.resnet34(pretrained=False, progress=False)
    print(resnet)
