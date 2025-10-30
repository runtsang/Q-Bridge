"""Enhanced classical QCNN model with hierarchical branching and dropout regularization."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class QCNNEnhanced(nn.Module):
    """
    Classical QCNN-inspired network with two parallel branches and dropout.
    The branch with Tanh emulates the original architecture, while the
    branch with ReLU introduces an alternative non‑linearity. The outputs
    are combined by averaging before the final sigmoid head.
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        # Feature map
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())

        # Branch 1 – uses Tanh activations
        self.branch1 = nn.Sequential(
            nn.Linear(16, 16), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(16, 12), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(12, 8), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(8, 4), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(4, 4), nn.Tanh(), nn.Dropout(dropout),
        )

        # Branch 2 – uses ReLU activations
        self.branch2 = nn.Sequential(
            nn.Linear(16, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 12), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(12, 8), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(8, 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4, 4), nn.ReLU(), nn.Dropout(dropout),
        )

        # Final classification head
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = (out1 + out2) / 2.0
        return torch.sigmoid(self.head(out))


def build_QCNNEnhanced(dropout: float = 0.2) -> QCNNEnhanced:
    """Factory function returning a configured :class:`QCNNEnhanced`."""
    return QCNNEnhanced(dropout=dropout)


__all__ = ["QCNNEnhanced", "build_QCNNEnhanced"]
