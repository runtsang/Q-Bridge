"""Enhanced classical QCNN with dropout, batch‑norm and residual connections."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QCNNGen168Model(nn.Module):
    """
    A deeper, regularised convolution‑inspired network.
    - Uses BatchNorm1d after each linear layer.
    - Applies ReLU activations followed by Dropout.
    - Final sigmoid output for binary classification.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: list[int] | tuple[int,...] = (16, 32, 16),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_extractor(x)
        return torch.sigmoid(self.classifier(x))


def QCNNGen168() -> QCNNGen168Model:
    """Factory returning a fully‑configured :class:`QCNNGen168Model` instance."""
    return QCNNGen168Model()


__all__ = ["QCNNGen168", "QCNNGen168Model"]
