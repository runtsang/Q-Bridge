from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridBinaryRegressionNet(nn.Module):
    """
    Classical hybrid network for binary classification or regression.
    The network consists of a residual CNN backbone followed by a dense
    head that optionally applies a learnable shift before a sigmoid
    activation.  The class is fully compatible with the original
    ``ClassicalQuantumBinaryClassification`` model but adds
    dropout, batch‑norm and residual connections for better
    generalisation.
    """

    def __init__(
        self,
        input_channels: int = 3,
        base_filters: int = 32,
        dropout: float = 0.3,
        shift: float = 0.0,
        classification: bool = True,
    ) -> None:
        super().__init__()
        self.classification = classification
        self.shift = shift
        # Residual block 1
        self.conv1 = nn.Conv2d(input_channels, base_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.conv1_res = nn.Conv2d(input_channels, base_filters, kernel_size=1, stride=1, padding=0)
        # Residual block 2
        self.conv2 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)
        self.conv2_res = nn.Conv2d(base_filters, base_filters * 2, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout)
        # Dense head
        self.fc1 = nn.Linear((base_filters * 2) * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, classification: bool | None = None) -> torch.Tensor:
        if classification is None:
            classification = self.classification
        # Residual block 1
        res = self.conv1_res(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = x + res
        x = self.drop(x)
        # Residual block 2
        res = self.conv2_res(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x + res
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        if classification:
            probs = self.sigmoid(x + self.shift)
            return torch.cat((probs, 1 - probs), dim=-1)
        else:
            return x.squeeze(-1)

__all__ = ["HybridBinaryRegressionNet"]
