"""Hybrid classical model integrating convolutional feature extractor and a quantum‑inspired filter.

The class `QuantumNATHybrid` inherits from `torch.nn.Module` and implements a deeper CNN
architecture with a residual block.  A lightweight classical filter emulates the
`Conv` routine from the original `Conv.py` seed and its output is concatenated with
the flattened CNN features before the fully‑connected head.  Batch‑normalisation is
applied to the final 4‑dimensional output.

The design keeps the classical scaling of the convolutional backbone while adding
a tunable thresholded filter that mimics the behaviour of the quantum filter,
allowing downstream experiments to swap in a true quantum module without
modifying the rest of the network.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """Thresholded 2‑D convolution acting as a drop‑in quantum‑filter emulator."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        return torch.sigmoid(logits - self.threshold).mean(dim=[1, 2, 3], keepdim=True)

class ResidualBlock(nn.Module):
    """Simple residual block used in the backbone."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class QuantumNATHybrid(nn.Module):
    """Hybrid classical model combining CNN, residuals and a quantum‑filter emulator."""
    def __init__(self) -> None:
        super().__init__()
        # Backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResidualBlock(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classical filter emulating quantum behaviour
        self.filter = ConvFilter(kernel_size=2, threshold=0.0)
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x).view(bsz, -1)
        filt = self.filter(x)  # shape (bsz, 1)
        combined = torch.cat([feat, filt], dim=1)
        logits = self.classifier(combined)
        return self.norm(logits)

__all__ = ["QuantumNATHybrid"]
