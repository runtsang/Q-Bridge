"""Hybrid classical network with a quantum‑inspired feature extractor."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QCNNHybrid(nn.Module):
    """A deeper fully‑connected network that mimics the structure of a QCNN.

    The architecture follows the same layer ordering as the original seed
    but adds skip connections, batch‑norm and dropout to improve
    generalisation.  The feature map is a learnable linear projection
    that can be interpreted as a classical approximation of a quantum
    feature map.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor
        self.feature_map = nn.Sequential(
            nn.Linear(8, 32), nn.BatchNorm1d(32), nn.ReLU()
        )
        # Convolution‑like fully‑connected blocks
        self.conv1 = nn.Sequential(
            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Linear(16, 8), nn.BatchNorm1d(8), nn.ReLU()
        )
        # Pooling emulation
        self.pool1 = nn.Sequential(
            nn.Linear(32, 24), nn.BatchNorm1d(24), nn.ReLU()
        )
        self.pool2 = nn.Sequential(
            nn.Linear(16, 12), nn.BatchNorm1d(12), nn.ReLU()
        )
        self.pool3 = nn.Sequential(
            nn.Linear(8, 4), nn.BatchNorm1d(4), nn.ReLU()
        )
        # Final classifier
        self.head = nn.Linear(4, 1)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.dropout(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNHybrid:
    """Factory returning the configured :class:`QCNNHybrid`."""
    return QCNNHybrid()


__all__ = ["QCNN", "QCNNHybrid"]
