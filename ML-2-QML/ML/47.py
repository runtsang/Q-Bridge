"""HybridQuantumClassifier: classical backbone + classical head for binary classification.

This module extends the original seed by adding residual blocks, batch‑norm,
and a focal‑loss helper. It keeps the public API identical to the original
`HybridQuantumClassifier`. The class is fully classical and can be used as a
drop‑in replacement for the quantum version in training pipelines that
require only CPU/GPU resources.

API:
    HybridQuantumClassifier(num_classes=2, device='cpu')
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HybridQuantumClassifier", "ResidualBlock", "FocalLoss"]

class ResidualBlock(nn.Module):
    """Simple 2‑D residual block with batch‑norm."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)

class FocalLoss(nn.Module):
    """Focal loss for binary classification."""
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = -((1 - pt) ** self.gamma) * torch.log(pt + 1e-12)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class HybridQuantumClassifier(nn.Module):
    """Classical backbone + classical head for binary classification."""
    def __init__(self, num_classes: int = 2, device: str = "cpu"):
        super().__init__()
        self.device = device
        # Backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.MaxPool2d(2),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        # Linear classifier
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)
        self.loss_fn = nn.BCELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        probs = torch.sigmoid(logits)
        return probs

    def compute_loss(self, probs: torch.Tensor, targets: torch.Tensor,
                     use_focal: bool = False, gamma: float = 2.0) -> torch.Tensor:
        if use_focal:
            focal = FocalLoss(gamma=gamma)
            return focal(probs, targets)
        else:
            return self.loss_fn(probs, targets)
