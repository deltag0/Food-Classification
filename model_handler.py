import torch
import torchmetrics
import os
from torch import nn
from pathlib import Path

from constants import EPOCHS, DEVICE, BATCH_SIZE
from torch.utils.data import DataLoader


def test_loop(model: torch.nn.Module, loss_function: torch.nn.Module, loader: DataLoader,
              loader_len: int, device=DEVICE):
    """
    test_loops uses data that wasn't used in the model's training to evaluate the performance of the model
    when it comes to data the model hasn't seen. Same process as the training loop minus the need for updating 
    the weights.
    """

    overall_acc = 0
    overall_loss = 0
    acc_fn = torchmetrics.Accuracy('multiclass', num_classes=3)

    model.eval()
    with torch.inference_mode():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            test_logits = model(X)
            test_preds = torch.argmax(torch.softmax(test_logits, 1), 1)

            overall_acc += acc_fn(test_preds, y).item()
            loss = loss_function(test_logits, y).item()
            overall_loss += loss

        overall_loss /= loader_len
        overall_acc /= loader_len
        overall_acc *= 100

    print(f"Model got an accuracy of {overall_acc} with a loss of {overall_loss}")

    return (overall_loss, overall_acc)


def train_loop(model: torch.nn.Module, optimizer: torch.nn.Module, loss_function: torch.nn.Module,
                loader: DataLoader, loader_len: int, device=DEVICE) -> tuple[float, float]:
    """
    train_loop is loops through all the elements in the loader, doing the forward pass,
    finding the loss, doing back propagation and updates the weights of the model.
    """

    model.train()
    overall_loss = 0
    overall_acc = 0
    acc_fn = torchmetrics.Accuracy('multiclass', num_classes=3)

    # batches of size s, so the shape of of an element in loader is (s x 3 x 227 x 227)
    for batch, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        train_logits = model(X)
        train_preds = torch.argmax(torch.softmax(train_logits, 1), 1)

        overall_acc += acc_fn(train_preds, y)
        loss = loss_function(train_logits, y)
        overall_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        if batch % 4 == 0:
            print(f"Evaluated {BATCH_SIZE * (batch + 1)} / {BATCH_SIZE * loader_len} samples")

        overall_loss /= loader_len
        overall_acc /= loader_len
        overall_acc *= 100

        print(f"Train loss: {overall_loss}")

    return (overall_loss, overall_acc)


def handler(model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: torch.nn.Module, trainer_loader: DataLoader,
                  train_loader_len: int, test_loader: torch.nn.Module, test_loader_len: int, epochs: int=EPOCHS,) -> dict[str:float]:
    """
    handler handles the training and testing of the model, as well as summarizing it's training and validation report.
    Losses and accuracies of the training and validation process are stored in the mode_data dict to be returned.

    If using the scheduler, we need to add scheduler.step() for each epoch
    """
    model_data = {
        "test_loss": [],
        "test_acc": [],
        "train_loss": [],
        "train_acc": []
    }

    save_name = "test_modelV2.pth"

    for epoch in range(epochs):
        train_loss, train_acc = train_loop(model, optimizer, loss_fn, trainer_loader, train_loader_len)
        test_loss, test_acc = test_loop(model, loss_fn, test_loader, test_loader_len)

        print("========================================")

        model_data['test_loss'].append(test_loss)
        model_data['test_acc'].append(test_acc)
        model_data['train_loss'].append(train_loss)
        model_data['train_acc'].append(train_acc)

    models_path = Path(os.path.dirname(os.path.abspath(__file__)) + "\\trained_model\\" + save_name)
    torch.save(model.state_dict(), models_path)

    return model_data
