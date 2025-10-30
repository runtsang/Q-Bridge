"""Hybrid kernel method combining classical RBF kernel with convolutional feature extraction.

The module defines a shared class :class:`HybridKernel` that can be used for classical
experiments. It internally uses a 2×2 convolutional filter (mimicking the
``Quanvolution`` pattern) followed by a radial‑basis‑function kernel on the flattened
feature maps.  An alias ``Kernel`` is provided for backward compatibility with
the original ``QuantumKernelMethod`` interface.

The class exposes a ``forward`` method that returns the kernel value between two
inputs and a ``kernel_matrix`` helper for Gram‑matrix construction.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ClassicalQuanvolutionFilter(nn.Module):
    """
    Classical 2×2 convolutional filter that extracts local patches from a
    single‑channel image and flattens them into a feature vector.
    """
    def __init__(self) -> None:
        super().__init__()
        # 4 output channels correspond to the 4 elements of a 2×2 patch
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x shape: (batch, 1, H, W)
        features = self.conv(x)  # (batch, 4, H/2, W/2)
        return features.view(x.size(0), -1)  # (batch, 4 * H/2 * W/2)


class KernalAnsatz(nn.Module):
    """Classical RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class HybridKernel(nn.Module):
    """
    Combines a classical convolutional feature extractor with an RBF kernel.
    The interface mirrors the original ``Kernel`` class but adds a convolution
    stage.  It can be used as a drop‑in replacement in classical pipelines.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.feature_extractor = ClassicalQuanvolutionFilter()
        self.kernel = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value between two batches of images.
        """
        # Extract features
        feat_x = self.feature_extractor(x)
        feat_y = self.feature_extractor(y)
        # Compute RBF kernel on flattened feature vectors
        return self.kernel(feat_x, feat_y).squeeze()

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """
        Return Gram matrix for two batches of images.
        """
        return np.array([[self.forward(a[i:i+1], b[j:j+1]).item() for j in range(b.size(0))] for i in range(a.size(0))])


# Backward compatibility alias
Kernel = HybridKernel

__all__ = ["HybridKernel", "Kernel", "KernalAnsatz"]
