"""Hybrid classical QCNN that mimics quantum convolutional structure while adding a learnable RBF kernel and regression head."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RBFKernelLayer(nn.Module):
    """Applies an RBF kernel to the input features and projects them onto learnable weights."""
    def __init__(self, gamma: float = 1.0, out_features: int = 64) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        diff = x.unsqueeze(1) - x.unsqueeze(0)          # (batch, batch, features)
        dist_sq = (diff ** 2).sum(-1)                   # (batch, batch)
        K = torch.exp(-self.gamma * dist_sq)            # (batch, batch)
        return K @ self.weight                         # (batch, out_features)


class EstimatorNN(nn.Module):
    """Simple regression/classification head."""
    def __init__(self, in_features: int, out_features: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.Tanh(),
            nn.Linear(32, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridQCNN(nn.Module):
    """
    Classical QCNN that uses 2×2 patches (like Quanvolution), a learnable RBF kernel,
    and a regression head. Mirrors the structure of the original QCNN while adding
    kernel‑based feature enrichment.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, gamma: float = 1.0) -> None:
        super().__init__()
        # 2×2 patches with stride 2 -> similar to QuanvolutionFilter
        self.patch_conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2)
        # Flatten to (batch, 4*14*14)
        self.kernel_layer = RBFKernelLayer(gamma=gamma, out_features=64)
        # Final classification/regression head
        self.classifier = EstimatorNN(in_features=64, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        patches = self.patch_conv(x)                     # (batch, 4, 14, 14)
        features = patches.view(patches.size(0), -1)      # (batch, 4*14*14)
        kernel_features = self.kernel_layer(features)    # (batch, 64)
        logits = self.classifier(kernel_features)        # (batch, num_classes)
        return logits


def HybridQCNNFactory(num_classes: int = 10, gamma: float = 1.0) -> HybridQCNN:
    """Convenience factory mirroring the original QCNN() API."""
    return HybridQCNN(num_classes=num_classes, gamma=gamma)


__all__ = ["HybridQCNN", "HybridQCNNFactory", "RBFKernelLayer", "EstimatorNN"]
