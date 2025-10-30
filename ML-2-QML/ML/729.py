"""Hybrid classical model that extends the original QuantumNAT with residual connections and deeper feature extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumNATHybrid(nn.Module):
    """Classical CNN with residual connection and fully connected head."""

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor: two conv layers + pooling
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Residual mapping from input to feature space
        self.residual = nn.Conv2d(1, 16, kernel_size=1, stride=4, padding=0)
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute residual
        res = self.residual(x)
        feat = self.features(x)
        # Add residual
        out = feat + res
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.norm(out)


__all__ = ["QuantumNATHybrid"]
