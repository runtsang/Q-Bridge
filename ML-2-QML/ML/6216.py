"""Classical hybrid model inspired by Quantum‑NAT and binary classification.

The network consists of a convolutional feature extractor followed by a
fully‑connected head.  A differentiable sigmoid activation is used to
mirror the quantum expectation head of the reference QML implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridSigmoid(nn.Module):
    """Classical sigmoid head with an optional shift.

    The shift parameter is useful when the downstream quantum circuit
    expects inputs in a specific range.
    """
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(inputs + self.shift)


class QuantumNATHybrid(nn.Module):
    """Classical convolutional network that mimics the Quantum‑NAT architecture.

    The architecture follows the pattern of the original Quantum‑NAT
    seed but replaces the quantum layer with a lightweight classical
    sigmoid head.  The design keeps feature extraction, batch‑norm
    and pooling identical to the QML version so that the two variants
    can be benchmarked side‑by‑side.
    """
    def __init__(self, in_channels: int = 1, n_classes: int = 4) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Fully‑connected projector
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )
        self.norm = nn.BatchNorm1d(n_classes)
        self.head = HybridSigmoid()  # mimic quantum expectation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # feature extraction
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.norm(x)
        # sigmoid head
        return self.head(x)


__all__ = ["QuantumNATHybrid"]
