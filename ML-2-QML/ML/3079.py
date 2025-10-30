"""Hybrid kernel combining convolutional feature extraction and RBF kernel.

This module defines :class:`HybridKernelQCNN` that first transforms inputs with a
convolution-inspired feature extractor (mirroring the classical QCNNModel) and
then evaluates a radial‑basis‑function kernel on the extracted features.  The
class is fully compatible with the original ``QuantumKernelMethod`` API and
can be used as a drop‑in replacement for the classical kernel.

The implementation reuses the architecture from the classical QCNN seed:
linear layers with tanh activations and a final sigmoid output.  The RBF
kernel is applied to the 4‑dimensional bottleneck representation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class HybridKernelQCNN(nn.Module):
    """Convolutional feature extractor + RBF kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        # Feature extractor mimicking QCNNModel
        self.feature_extractor = nn.Sequential(
            nn.Linear(8, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh()
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Extract features
        fx = self.feature_extractor(x)
        fy = self.feature_extractor(y)
        # Compute RBF kernel on extracted features
        diff = fx - fy
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix using :class:`HybridKernelQCNN`."""
    kernel = HybridKernelQCNN(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["HybridKernelQCNN", "kernel_matrix"]
