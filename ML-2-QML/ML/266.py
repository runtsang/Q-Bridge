"""
Classical QCNN with configurable depth, dropout, and batch‑norm.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Iterable, Tuple


class QCNNModel(nn.Module):
    """
    A fully‑connected network that mimics the structure of a QCNN.
    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features (default 8).
    hidden_dims : Iterable[int]
        Sequence of hidden layer sizes. Defaults to (16, 16, 12, 8, 4, 4).
    dropout : float
        Dropout probability applied after each hidden layer.
    batch_norm : bool
        Whether to insert a BatchNorm1d after each hidden layer.
    """
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: Iterable[int] | None = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = (16, 16, 12, 8, 4, 4)
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))

    # ------------------------------------------------------------------ #
    # Training utilities
    # ------------------------------------------------------------------ #
    def train_epoch(
        self,
        data_loader: DataLoader,
        optimizer: AdamW,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        Returns mean loss and accuracy on the epoch data.
        """
        self.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        criterion = nn.BCELoss()
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()
            optimizer.zero_grad()
            preds = self(batch_x).squeeze()
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            preds_bin = (preds > 0.5).float()
            total_correct += (preds_bin == batch_y).sum().item()
            total_samples += batch_x.size(0)
        return total_loss / total_samples, total_correct / total_samples

    def evaluate(
        self,
        data_loader: DataLoader,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Evaluate the model on a validation/test set.
        Returns mean loss and accuracy.
        """
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        criterion = nn.BCELoss()
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float()
                preds = self(batch_x).squeeze()
                loss = criterion(preds, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                preds_bin = (preds > 0.5).float()
                total_correct += (preds_bin == batch_y).sum().item()
                total_samples += batch_x.size(0)
        return total_loss / total_samples, total_correct / total_samples


__all__ = ["QCNNModel"]
