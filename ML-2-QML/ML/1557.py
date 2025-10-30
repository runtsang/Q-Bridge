"""Enhanced classical model for Quantum‑NAT experiments.

The class keeps the same functional signature as the original QFCModel
but introduces a deeper convolutional backbone, residual connections,
dropout, and a more expressive fully‑connected head.  This allows
benchmarking against a richer classical baseline without changing any
training pipelines that expect a 4‑dimensional output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumNATModel(nn.Module):
    """Convolutional‑dense network with residual skip connections and dropout.

    Parameters
    ----------
    dropout : float, optional
        Dropout probability applied after each dense block.  Defaults to 0.3.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        # Conv‑residual blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dense head
        self.fc1 = nn.Linear(32, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(64, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Residual connections
        residual = x
        x = self.conv_block1(x)
        x = x + residual
        residual = x
        x = self.conv_block2(x)
        x = x + residual
        residual = x
        x = self.conv_block3(x)
        x = x + residual

        x = self.pool(x).view(x.size(0), -1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        out = self.out(x)
        return self.norm(out)


__all__ = ["QuantumNATModel"]
