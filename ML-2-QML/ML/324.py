"""Extended classical QCNN architecture with residuals and dropout."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class QCNNModel(nn.Module):
    """
    A deeper, regularised QCNN‑style network.

    The model mimics a convolution‑pool‑convolution pipeline but replaces
    each block with a fully‑connected layer followed by BatchNorm,
    ReLU, and optional dropout.  Residual connections are added after
    each convolution‑pool pair to preserve feature information and
    accelerate training.
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout)
        )
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout)
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12), nn.BatchNorm1d(12), nn.ReLU(), nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(dropout)
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Dropout(dropout)
        )
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Feature extraction
        f = self.feature_map(x)

        # Residual block 1
        r1 = self.conv1(f)
        r1 = self.pool1(r1)
        f = F.relu(r1 + f)

        # Residual block 2
        r2 = self.conv2(f)
        r2 = self.pool2(r2)
        f = F.relu(r2 + f)

        # Final convolution
        f = self.conv3(f)

        return torch.sigmoid(self.head(f))


def QCNN(dropout: float = 0.1) -> QCNNModel:
    """Return a configured :class:`QCNNModel`."""
    return QCNNModel(dropout=dropout)


__all__ = ["QCNNModel", "QCNN"]
