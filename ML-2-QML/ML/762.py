"""Enhanced classical QCNN with residuals, batch‑norm, and dropout."""

from __future__ import annotations

import torch
from torch import nn


class QCNNExtendedModel(nn.Module):
    """
    Fully‑connected network that mimics a QCNN but with modern regularisation.

    Layers:
        * Feature map : Linear → Tanh
        * Residual block 1 : Linear → BatchNorm → Dropout → Linear → BatchNorm → Add
        * Residual block 2 : Linear → BatchNorm → Dropout → Linear → BatchNorm → Add
        * Final head : Linear → Sigmoid
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        # Residual block 1
        self.res1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
        )
        # Residual block 2
        self.res2 = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.Dropout(dropout),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
        )
        self.head = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        # Residual 1
        res = self.res1(x)
        x = nn.functional.relu(x + res)
        # Residual 2
        res = self.res2(x)
        x = nn.functional.relu(x + res)
        return torch.sigmoid(self.head(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return binary predictions (0 or 1) for a batch of inputs.
        """
        with torch.no_grad():
            probs = self(x)
        return (probs > 0.5).long()


def QCNNExtended() -> QCNNExtendedModel:
    """Factory returning the configured :class:`QCNNExtendedModel`."""
    return QCNNExtendedModel()


__all__ = ["QCNNExtended", "QCNNExtendedModel"]
