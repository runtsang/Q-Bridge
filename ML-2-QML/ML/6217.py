"""Classical hybrid classifier that fuses CNN feature extraction with a learnable RBF kernel head.

The module implements a PyTorch model that:
* Extracts visual features with a lightweight convolutional backbone.
* Projects the features into a high‑dimensional kernel space using a set of learnable support vectors.
* Learns a linear decision boundary on the kernel features for binary classification.

This design preserves the interpretability of kernel methods while leveraging deep feature learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RBFKernelLayer(nn.Module):
    """Learnable RBF kernel layer with support‑vector encoding."""
    def __init__(self, in_features: int, num_support: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.support = nn.Parameter(torch.randn(num_support, in_features))
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, features]
        diff = x.unsqueeze(1) - self.support.unsqueeze(0)  # [batch, num_support, features]
        dist_sq = (diff ** 2).sum(-1)
        return torch.exp(-self.gamma * dist_sq)  # [batch, num_support]


class HybridKernelClassifier(nn.Module):
    """CNN backbone followed by a learnable RBF kernel head and linear output."""
    def __init__(
        self,
        in_channels: int = 3,
        num_support: int = 20,
        gamma: float = 1.0,
        hidden_dim: int = 120,
        kernel_dim: int = 84,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Dropout2d(0.2),
            nn.Flatten(),
        )
        # Determine feature dimension
        dummy = torch.zeros(1, in_channels, 32, 32)
        feat_dim = self.backbone(dummy).shape[1]
        self.kernel_layer = RBFKernelLayer(feat_dim, num_support, gamma)
        self.classifier = nn.Linear(num_support, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        kernel_features = self.kernel_layer(features)  # [batch, num_support]
        logits = self.classifier(kernel_features).squeeze(-1)
        probs = torch.sigmoid(logits)
        return torch.cat([probs.unsqueeze(-1), (1 - probs).unsqueeze(-1)], dim=-1)


__all__ = ["HybridKernelClassifier", "RBFKernelLayer"]
