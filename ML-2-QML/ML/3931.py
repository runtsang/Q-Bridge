"""Hybrid classical model combining convolutional feature extraction with a quantum-inspired transformation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumInspiredTransform(nn.Module):
    """Classical layer mimicking quantum feature maps via orthogonal transforms and non-linear phases."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.orthogonal_(self.linear.weight)
        self.phase = nn.Parameter(torch.randn(out_features) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return torch.sin(x + self.phase)


class QFCModel(nn.Module):
    """Hybrid CNN + fully connected + quantum-inspired transform."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.AdaptiveAvgPool2d(1),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.qsim = QuantumInspiredTransform(64, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.qsim(x)
        return self.norm(x)


__all__ = ["QFCModel"]
