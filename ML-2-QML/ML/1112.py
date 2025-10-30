"""
Enhanced classical QCNN model with residual connections, batch‑norm,
dropout, and a flexible training interface.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Iterable, Tuple


class ResidualBlock(nn.Module):
    """
    A single residual block consisting of a linear layer, batch‑norm,
    ReLU, dropout and a skip connection.  If the input and output
    dimensions differ, a linear projection is applied to the skip
    connection.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out + self.skip(x)


class QCNNModel(nn.Module):
    """
    A stack of residual fully‑connected layers that emulates the
    quantum convolution steps of the original QCNN.  The network is
    fully parameterised and can be trained with any PyTorch optimiser.
    """
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: Iterable[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 12, 8, 4, 4]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(ResidualBlock(prev, h, dropout))
            prev = h
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = torch.sigmoid(self.head(x))
        return x

    def train_loop(
        self,
        dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 50,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        """
        Very small training loop that demonstrates how the model can be
        trained on a PyTorch DataLoader.  The loop uses BCELoss and Adam.
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
                optimizer.zero_grad()
                preds = self(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss/len(dataloader):.4f}")


def QCNN() -> QCNNModel:
    """
    Factory returning a ready‑to‑train QCNNModel instance.
    """
    return QCNNModel()


__all__ = ["QCNNModel", "QCNN"]
