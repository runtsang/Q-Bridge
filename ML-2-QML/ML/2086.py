"""Classical QCNNGen299 model with enhanced regularisation and skip‑like connections."""
from __future__ import annotations

import torch
from torch import nn


class QCNNGen299Model(nn.Module):
    """A dense network mimicking a QCNN with residual‑style skip connections and dropout."""

    def __init__(self) -> None:
        super().__init__()
        # Feature extraction
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        # Convolution‑pool stages with skip‑like connections
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.BatchNorm1d(12),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 12),
            nn.BatchNorm1d(12),
            nn.ReLU()
        )
        self.pool2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        self.pool3 = nn.Sequential(
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU()
        )
        # Output head
        self.head = nn.Linear(4, 1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.pool3(x)
        return torch.sigmoid(self.head(x))


def QCNNGen299() -> QCNNGen299Model:
    """Factory for a QCNNGen299Model."""
    return QCNNGen299Model()


__all__ = ["QCNNGen299", "QCNNGen299Model"]
