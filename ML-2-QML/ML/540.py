"""
Extended classical QCNN model with dropout, batch‑norm and residual connections.
"""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

__all__ = ["QCNNModel", "QCNN", "train_qcnn", "evaluate_qcnn"]


class QCNNModel(nn.Module):
    """
    A lightweight QCNN‑inspired feed‑forward network.

    The architecture mirrors the original seed but adds:
    * Dropout after each convolutional block.
    * Batch‑normalisation before non‑linearity.
    * A residual skip connection between the first and third convolutional blocks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16), nn.BatchNorm1d(16), nn.Tanh()
        )
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.Tanh()
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12), nn.BatchNorm1d(12), nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8), nn.BatchNorm1d(8), nn.Tanh()
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4), nn.BatchNorm1d(4), nn.Tanh()
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Tanh()
        )
        self.dropout = nn.Dropout(p=0.3)
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        # Residual connection
        residual = x
        x = self.conv2(x)
        x = self.pool2(x)
        # Combine with residual
        x = x + residual
        x = self.conv3(x)
        x = self.dropout(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Factory returning a fresh :class:`QCNNModel`."""
    return QCNNModel()


def train_qcnn(
    model: QCNNModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """
    Trains the QCNNModel with binary cross‑entropy loss.

    Parameters
    ----------
    model : QCNNModel
        The neural network to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    device : str
        Device to run the model on.

    Returns
    -------
    history : dict
        History of training and validation losses.
    """
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                preds = model(xb).squeeze()
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:02d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

    return history


def evaluate_qcnn(
    model: QCNNModel,
    test_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """
    Computes the binary accuracy on the test set.

    Parameters
    ----------
    model : QCNNModel
        Trained model.
    test_loader : DataLoader
        Test data loader.
    device : str
        Device to run the model on.

    Returns
    -------
    accuracy : float
        Accuracy in the range [0, 1].
    """
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            preds = model(xb).squeeze()
            predicted = (preds > 0.5).float()
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
    return correct / total
