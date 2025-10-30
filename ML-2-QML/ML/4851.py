"""Hybrid classical CNN+FC model with optional RBF kernel.

The model combines a convolutional backbone with a fully connected head
and exposes a convenient RBF kernel interface for pairwise similarity
computations.  It can be used directly as a classifier or as a feature
extractor for downstream kernel methods.
"""

import numpy as np
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class QFCModel(nn.Module):
    """CNN + FC backbone with RBF kernel support."""

    def __init__(self, num_classes: int = 4, gamma: float = 1.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN+FC pipeline."""
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the Gaussian RBF kernel between two tensors."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix for datasets ``a`` and ``b``."""
        return np.array(
            [[self.rbf_kernel(x, y).item() for y in b] for x in a]
        )
