"""Hybrid classical model combining a CNN, fully connected layers, and a radial basis function kernel head for classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class QFCModel(nn.Module):
    """
    Classical hybrid model:
    - CNN feature extractor
    - Optional RBF kernel-based classification head with learnable prototypes
    - Supports batch normalization and flexible feature dimensionality
    """

    def __init__(self, num_classes: int = 4, kernel_gamma: float = 1.0, use_kernel: bool = True):
        super().__init__()
        self.use_kernel = use_kernel
        self.kernel_gamma = kernel_gamma

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Dummy forward to infer feature dimension
        dummy = torch.zeros(1, 1, 28, 28)
        feat_dim = self.features(dummy).view(1, -1).shape[1]

        # Fully connected projection
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feat_dim),
        )
        self.norm = nn.BatchNorm1d(feat_dim)

        # Kernel-based head
        if self.use_kernel:
            self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))
            self.prototypes_norm = nn.BatchNorm1d(num_classes)

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF kernel matrix between two batches.
        """
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (B,1,D)-(1,N,D)
        dist_sq = (diff * diff).sum(dim=2)  # (B,N)
        return torch.exp(-self.kernel_gamma * dist_sq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing logits for classification.
        """
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        out = self.norm(out)

        if self.use_kernel:
            sims = self._rbf_kernel(out, self.prototypes)  # (B, num_classes)
            sims = self.prototypes_norm(sims)
            logits = sims
        else:
            logits = out

        return logits
