"""QCNNGen167 – a modern, extensible classical QCNN implementation."""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Iterable


class QCNNLayer(nn.Module):
    """A single convolution‑pooling block with optional dropout and batchnorm."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.tanh(self.bn(self.conv(x))))


class QCNNModel(nn.Module):
    """Stack of QCNNLayer blocks ending with a sigmoid head."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                QCNNLayer(8, 16, dropout),
                QCNNLayer(16, 12, dropout),
                QCNNLayer(12, 8, dropout),
                QCNNLayer(8, 4, dropout),
            ]
        )
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return torch.sigmoid(self.head(x))


def QCNN(dropout: float = 0.0) -> QCNNModel:
    """Factory for the configurable QCNNModel."""
    return QCNNModel(dropout)


def train(
    model: nn.Module,
    data: Tuple[torch.Tensor, torch.Tensor],
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
) -> Iterable[float]:
    """Yield training loss per epoch."""
    model = model.to(device)
    dataset = TensorDataset(data[0], data[1])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x).squeeze()
            loss = criterion(output, y.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        yield epoch_loss / len(loader)


__all__ = ["QCNN", "QCNNModel", "train"]
