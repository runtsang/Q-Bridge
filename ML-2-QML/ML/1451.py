"""Hybrid classical model for Quantum‑NAT with advanced feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumNATHybrid(nn.Module):
    """Classical hybrid model with CNN backbone, learnable pooling, and meta‑classifier."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()
        # CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Learnable global pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected head
        self.head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.backbone(x)
        pooled = self.pool(features).view(x.size(0), -1)
        logits = self.head(pooled)
        return self.norm(logits)


__all__ = ["QuantumNATHybrid"]
