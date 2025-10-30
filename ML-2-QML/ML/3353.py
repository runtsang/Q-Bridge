"""Hybrid classical classifier with optional RBF kernel.

This module builds a reusable class that mirrors the quantum interface
while exposing a richer set of features.  The design follows the
original ``QuantumClassifierModel`` seed but adds:
* a depth‑controlled feed‑forward backbone,
* a configurable RBF kernel (classical),
* a training‑ready ``forward`` method that returns logits and the kernel
  matrix for use in downstream SVM or kernel‑based loss functions.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import numpy as np


class _RBFKernel:
    """Internal helper that supports both classical and quantum kernels."""
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Compute the RBF kernel between two batches of samples
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # shape (Bx, By, D)
        dist_sq = torch.sum(diff * diff, dim=-1)  # shape (Bx, By)
        return torch.exp(-self.gamma * dist_sq)


class HybridClassifier(nn.Module):
    """Re‑defined classical classifier with optional kernel augmentation."""
    def __init__(
        self,
        num_features: int,
        depth: int = 2,
        use_kernel: bool = False,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_kernel = use_kernel
        # Build feed‑forward backbone
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.backbone = nn.Sequential(*layers)
        # Optional kernel
        self.kernel = _RBFKernel(gamma) if use_kernel else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits from the backbone."""
        return self.backbone(x)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Compute Gram matrix between two sets of samples."""
        if self.kernel is None:
            raise RuntimeError("Kernel is not enabled for this model.")
        kernel_tensor = self.kernel(a, b)
        return kernel_tensor.detach().cpu().numpy()

__all__ = ["HybridClassifier"]
