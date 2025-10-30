"""Enhanced classical model for QuantumNAT with residual connections and deeper CNN.

This module defines `QuantumNATEnhanced` which extends the original seed by adding:
- A three‑layer convolutional backbone with residual skip connections.
- A learnable batch‑norm layer after the fully‑connected head.
- Optional dropout for regularisation.
- A residual shortcut from the input image directly to the output feature vector.

The resulting network produces a 4‑dimensional embedding suitable for downstream tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """Three‑depth CNN + residual skip + fully‑connected projection."""

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2)

        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        self.out_norm = nn.BatchNorm1d(4)

        # Shortcut mapping from input to output
        self.shortcut_pool = nn.AdaptiveAvgPool2d(1)
        self.shortcut_fc = nn.Linear(1, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Residual shortcut: global avg pool and linear to 4 dims
        shortcut = self.shortcut_pool(x).view(x.size(0), -1)
        shortcut = self.shortcut_fc(shortcut)

        # Conv blocks with residual connections
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)

        res1 = out
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out += res1  # residual

        res2 = out
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.pool(out)
        out += res2  # residual

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.out_norm(out)

        # Combine with shortcut
        return out + shortcut

__all__ = ["QuantumNATEnhanced"]
