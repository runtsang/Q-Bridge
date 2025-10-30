"""Enhanced classical convolutional network for Quantum‑NAT experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATGen121(nn.Module):
    """
    A CNN backbone with residual connections, dropout, and batch normalization,
    followed by a fully connected head that outputs four features.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Residual block
        self.residual = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.dropout = nn.Dropout2d(p=0.25)
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 4)
        )
        self.out_norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        # Residual connection
        out = out + self.residual(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.out_norm(out)

__all__ = ["QuantumNATGen121"]
