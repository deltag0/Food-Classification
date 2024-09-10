import torch
import random
import os
from torch import nn
from pathlib import Path
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from constants import BATCH_SIZE, DIR, TRAIN_DIR, TEST_DIR, DEVICE, MODEL_NAME # constants
from models import AlexNet, VGGnet
from model_handler import handler, train_loop, test_loop
from visualizations import plot_loss_curves


if __name__ == "__main__":

    # check model in use (required to be AlexNet or VGGnet)
    if MODEL_NAME == "AlexNet":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    # transformation functions with data augmentation for the training dataset and weights from model

    train_transform = transforms.Compose([
        transforms.Resize([227, 227]),
        transforms.TrivialAugmentWide(31),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
        std=std,)
    ])
    test_tansform = transforms.Compose([
        transforms.Resize([227, 227]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
        std=std,)
    ])
    # Get functions as image folder from pytorch to ease data transformation

    train_info = datasets.ImageFolder(TRAIN_DIR, train_transform)
    test_info = datasets.ImageFolder(TEST_DIR, test_tansform)

    # get data as DataLoaders

    train_loader = DataLoader(
        train_info,
        BATCH_SIZE,
        True,
        num_workers=1
    )
    test_loader = DataLoader(
        test_info,
        BATCH_SIZE,
        False,
        num_workers=1
    )

    train_loader_len = len(train_loader)
    test_loader_len = len(test_loader)

    if MODEL_NAME == "AlexNet":
        model = AlexNet()
    else:
        model = VGGnet()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.005, weight_decay=0.005, momentum=0.9)
    # Optional paramater that could help if model is overfiting

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=15, gamma=0.5)

    model_data = handler(model, loss_fn, optimizer, train_loader,
                         train_loader_len, test_loader, test_loader_len)

    plot_loss_curves(model_data)
