"""Hybrid quanvolution filter with optional RBF kernel.

This module extends the original quanvolution filter by adding an
optional classical RBF kernel feature extractor.  The filter can be
used as a drop‑in replacement for the classical filter in the
original project, while the kernel option allows the model to learn
non‑linear relationships between patches.  The implementation is
fully NumPy/PyTorch and therefore compatible with any training
pipeline that relies on the original QuanvolutionFilter/QuanvolutionClassifier
classes.

The scaling paradigm is a *combination* of convolutional and kernel
methods, providing both local feature extraction and global similarity
measurements.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence


class QuanvolutionHybridFilter(nn.Module):
    """
    Classical quanvolution filter with optional RBF kernel feature map.

    Parameters
    ----------
    use_kernel : bool, default False
        If True, the filter outputs the RBF kernel between each 2×2 patch
        and a learned prototype vector.  This adds a non‑linear
        transformation to the patch representation.
    gamma : float, default 1.0
        RBF kernel width.  Only used when use_kernel is True.
    """

    def __init__(self, use_kernel: bool = False, gamma: float = 1.0) -> None:
        super().__init__()
        self.use_kernel = use_kernel
        self.gamma = gamma
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        if self.use_kernel:
            # Prototype vector for kernel computation
            self.register_buffer("prototype", torch.randn(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Feature vector of shape (B, 4 * 14 * 14).  When use_kernel is
            True the 4‑dimensional patch representation is replaced by
            an RBF kernel value against the prototype.
        """
        patches = self.conv(x)  # (B, 4, 14, 14)
        if self.use_kernel:
            # compute RBF kernel per patch
            B, C, H, W = patches.shape
            patches = patches.view(B, C, H * W)  # (B, 4, 196)
            proto = self.prototype.view(1, C, 1)
            diff = patches - proto
            k = torch.exp(-self.gamma * torch.sum(diff * diff, dim=1, keepdim=True))
            # k shape: (B, 1, 196) -> flatten to (B, 196)
            return k.view(B, -1)
        else:
            return patches.view(x.size(0), -1)


class QuanvolutionHybridClassifier(nn.Module):
    """
    Hybrid classifier that uses the QuanvolutionHybridFilter followed by
    a linear head.  The filter can be instantiated in classical mode
    (use_kernel=False) or with the kernel augmentation.

    Parameters
    ----------
    use_kernel : bool, default False
        Forward the filter with the kernel feature map.
    gamma : float, default 1.0
        RBF kernel width, forwarded to the filter.
    """

    def __init__(self, use_kernel: bool = False, gamma: float = 1.0) -> None:
        super().__init__()
        self.qfilter = QuanvolutionHybridFilter(use_kernel, gamma)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """
    Compute an RBF kernel Gram matrix between two sequences of tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of tensors to compare.  Each tensor is flattened
        before kernel evaluation.
    gamma : float, default 1.0
        Kernel width.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    def rbf(x, y):
        diff = x - y
        return torch.exp(-gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    return np.array([[rbf(x, y).item() for y in b] for x in a])


__all__ = ["QuanvolutionHybridFilter", "QuanvolutionHybridClassifier", "kernel_matrix"]
