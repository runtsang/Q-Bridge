"""
QCNNHybrid – classical implementation with residual layers and an optional
feature‑map encoder.  The model is fully Torch‑compatible and exposes a
`train` method that accepts data loaders and a loss function.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable, Iterable, Tuple


class ResidualBlock(nn.Module):
    """Two‑layer residual block with a skip connection."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        return self.activation(out)


class QCNNHybridModel(nn.Module):
    """
    Feature‑map encoder → residual convolution blocks → pooling → classifier.
    Designed to mirror the quantum QCNN topology while leveraging classical
    deep‑learning best practices.
    """
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: Iterable[int] | None = None,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [16, 12, 8, 4]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x)
        # Adaptive pooling expects a 3‑D tensor (batch, channels, seq)
        x = x.unsqueeze(-1)
        x = self.pool(x).squeeze(-1)
        logits = self.head(x)
        return torch.sigmoid(logits) if self.head.out_features == 1 else logits

    def train_loop(
        self,
        train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        val_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        epochs: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: str = "cpu",
    ) -> None:
        """End‑to‑end training routine with validation tracking."""
        self.to(device)
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                preds = self(x)
                loss = loss_fn(preds, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)

            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    preds = self(x)
                    val_loss += loss_fn(preds, y).item()
            val_loss /= len(val_loader)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train loss: {epoch_loss:.4f} | Val loss: {val_loss:.4f}"
            )


def QCNNHybrid(
    input_dim: int = 8,
    hidden_dims: Iterable[int] | None = None,
    num_classes: int = 1,
) -> QCNNHybridModel:
    """Factory returning a ready‑to‑train :class:`QCNNHybridModel`."""
    return QCNNHybridModel(input_dim, hidden_dims, num_classes)


__all__ = ["QCNNHybrid", "QCNNHybridModel"]
