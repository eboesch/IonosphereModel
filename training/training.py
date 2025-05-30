"""Training on a single epoch code."""

from tqdm import tqdm

# from typing import Any
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
import logging


def train_single_epoch(
    dataloader: DataLoader,
    model: Module,
    loss_fct: Module,  # nn.MSELoss, nn.L1Loss,
    optimizer: Optimizer,
    device: torch.device,
    logger: logging.Logger,
    log_interval: int,
) -> None:
    """
    Performs a single training epoch for the given model, using the given training data, loss function and optimizer.
    Every log_interval steps, the running loss is written to the logger.
    """

    # size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    running_loss = 0.0
    for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), mininterval=30):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        pred = model(X)
        loss = loss_fct(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # log statistics
        running_loss += loss.item()
        if batch % log_interval == log_interval - 1:
            logger.info(f"[{batch + 1:5d}/{len(dataloader):>5d}] loss: {running_loss / log_interval:.3f}")
            running_loss = 0.0
