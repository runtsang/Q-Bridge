"""Hybrid classical CNN + quantum‑inspired fully connected layer.

The model combines a convolutional feature extractor with a
parameterized fully connected block that mimics the behaviour
of the quantum layer in the original Quantum‑NAT reference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQFCModel(nn.Module):
    """Classical CNN front‑end followed by a quantum‑inspired FC block.

    The convolutional part is identical to the original
    Quantum‑NAT CNN, while the FC block replaces the quantum
    sub‑module with a learnable linear layer, a tanh non‑linearity
    and a batch normalisation.  This keeps the overall
    architecture compatible with the quantum version while
    remaining fully classical.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 16 channels × 7 × 7 feature map after two 2×2 pools
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.tanh = nn.Tanh()
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flat = features.view(bsz, -1)
        out = self.fc(flat)
        out = self.tanh(out)
        return self.norm(out)

__all__ = ["HybridQFCModel"]
