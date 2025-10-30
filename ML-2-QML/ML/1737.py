"""
QCNNGen323: An enhanced classical convolution‑inspired neural network.

The architecture expands upon the original seed by:
* Adding BatchNorm and Dropout layers for better regularisation.
* Introducing residual connections to ease gradient flow.
* Modelling the “pooling” step as learnable linear layers rather than simple dimensionality reductions.
"""

from __future__ import annotations

import torch
from torch import nn


class QCNNGen323Model(nn.Module):
    """ResNet‑style, batch‑normed, dropout‑augmented QCNN."""
    def __init__(self) -> None:
        super().__init__()

        # Feature map – first linear projection
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

        # Residual block 1
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        self.res1 = nn.Identity()

        # Pooling (dimensionality reduction) with learnable weight
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

        # Residual block 2
        self.conv2 = nn.Sequential(
            nn.Linear(12, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        self.res2 = nn.Identity()

        # Pooling 2
        self.pool2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

        # Final classifier
        self.fc = nn.Linear(8, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)

        # Residual block 1
        res = self.res1(x)
        x = self.conv1(x) + res
        x = torch.relu(x)

        x = self.pool1(x)

        # Residual block 2
        res = self.res2(x)
        x = self.conv2(x) + res
        x = torch.relu(x)

        x = self.pool2(x)

        return torch.sigmoid(self.fc(x))


def QCNNGen323() -> QCNNGen323Model:
    """Factory returning the configured QCNNGen323Model."""
    return QCNNGen323Model()


__all__ = ["QCNNGen323", "QCNNGen323Model"]
