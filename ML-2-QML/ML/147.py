"""Hybrid classical QCNN with residual‑style layers and training utilities."""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Iterable, Tuple


class QCNNHybrid(nn.Module):
    """
    A classical convolution‑inspired network with residual connections,
    batch‑normalisation, dropout and a sigmoid output for binary
    classification.  The architecture mirrors the original QCNN but
    uses fully‑connected layers that emulate quantum convolution steps.
    """

    def __init__(
        self,
        input_dim: int = 8,
        conv_dims: Tuple[int,...] = (16, 12, 8, 4, 4),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, conv_dims[0]),
            nn.BatchNorm1d(conv_dims[0]),
            nn.ReLU(),
        )

        conv_blocks = []
        for in_dim, out_dim in zip(conv_dims[:-1], conv_dims[1:]):
            conv_blocks.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.conv = nn.Sequential(*conv_blocks)

        self.head = nn.Linear(conv_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv(x)
        logits = self.head(x)
        return torch.sigmoid(logits)


def train_qcnn(
    model: nn.Module,
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """
    Train the QCNNHybrid model using binary cross‑entropy loss and Adam.

    Parameters
    ----------
    model : nn.Module
        The QCNNHybrid instance.
    data_loader : Iterable[Tuple[Tensor, Tensor]]
        Iterable yielding (inputs, targets) pairs.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    device : torch.device | str
        Target device.

    Returns
    -------
    nn.Module
        The trained model.
    """
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= len(data_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs} – Loss: {epoch_loss:.4f}")

    return model


def evaluate(
    model: nn.Module,
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str = "cpu",
) -> float:
    """
    Evaluate accuracy on a validation dataset.

    Parameters
    ----------
    model : nn.Module
        Trained QCNNHybrid instance.
    data_loader : Iterable[Tuple[Tensor, Tensor]]
        Iterable yielding (inputs, targets) pairs.
    device : torch.device | str
        Target device.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).round()
            correct += (preds.squeeze() == y).sum().item()
            total += y.size(0)
    return correct / total


__all__ = ["QCNNHybrid", "train_qcnn", "evaluate"]
