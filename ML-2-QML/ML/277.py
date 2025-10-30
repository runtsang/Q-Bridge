"""Extended classical QCNN with dropout, batch‑norm and configurable depth."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class QCNNHybrid(nn.Module):
    """QCNN‑inspired network with batch‑norm, dropout and depth control."""
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = dim
        self.feature = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        return torch.sigmoid(self.head(x))

    def freeze_layers(self, freeze: bool = True) -> None:
        """Freeze or unfreeze the feature extractor."""
        for param in self.feature.parameters():
            param.requires_grad = not freeze

    def summary(self) -> str:
        """Return a string representation of the model."""
        return str(self)


def QCNNHybridFactory() -> QCNNHybrid:
    """Convenient factory returning a fully configured instance."""
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "QCNNHybridFactory"]
