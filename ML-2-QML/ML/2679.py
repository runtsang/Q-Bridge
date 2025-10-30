"""Hybrid kernel combining classical RBF with QCNN feature extraction.

This module defines :class:`HybridKernelCNN`, a PyTorch implementation that
extracts features via a lightweight QCNN-like fully connected network and
applies an RBF kernel on those features.  It can be used as a drop‑in
replacement for the original `Kernel` class while offering richer
expressivity and a clear pathway to a quantum analogue.

The design mirrors the structure of the original `QuantumKernelMethod.py`
and `QCNN.py` seeds, but unifies them into a single, scalable interface.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Classical QCNN feature extractor (mirrors QCNNModel from seed)
# --------------------------------------------------------------------------- #
class _SimpleQCNNFeatureMap(nn.Module):
    """A lightweight, fully‑connected proxy for the quantum QCNN.

    The network emulates the structure of the seed QCNNModel: a feature map
    followed by three convolution‑like layers and two pooling stages.
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return x

# --------------------------------------------------------------------------- #
# Hybrid RBF kernel on QCNN features
# --------------------------------------------------------------------------- #
class HybridKernelCNN(nn.Module):
    """Hybrid classical kernel: QCNN feature extractor + RBF kernel.

    Parameters
    ----------
    gamma : float, optional
        RBF kernel width.  Default is ``1.0``.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.feature_extractor = _SimpleQCNNFeatureMap()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the hybrid kernel value for a pair of samples."""
        fx = self.feature_extractor(x)
        fy = self.feature_extractor(y)
        diff = fx - fy
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix between two datasets."""
        return np.array([[self.forward(torch.tensor(x), torch.tensor(y)).item() for y in b] for x in a])

__all__ = ["HybridKernelCNN"]
