"""Hybrid classical kernel combining RBF and optional convolutional feature extraction.

The class ``HybridKernel`` implements a batched RBF kernel that can optionally
pre‑process inputs with a small 2×2 convolutional filter.  The kernel is
fully differentiable and works with PyTorch tensors, making it suitable for
gradient‑based learning or kernel‑based algorithms such as SVMs.

The module also exposes a ``kernel_matrix`` helper that builds the Gram matrix
between two collections of feature vectors.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

class HybridKernel(nn.Module):
    """RBF kernel with optional convolutional feature extraction.

    Parameters
    ----------
    gamma : float, default 1.0
        RBF width parameter.
    use_conv : bool, default False
        If ``True`` a 2×2 stride‑2 convolution is applied before the RBF.
    conv_kernel_size : int, default 2
        Number of output channels for the convolution.  Only relevant if
        ``use_conv`` is ``True``.
    """
    def __init__(self, gamma: float = 1.0, use_conv: bool = False,
                 conv_kernel_size: int = 2) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_conv = use_conv
        if use_conv:
            # 1‑channel input → conv_kernel_size output channels
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=conv_kernel_size * conv_kernel_size,
                kernel_size=2,
                stride=2,
                bias=False
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Compute the RBF kernel ``k(x, y)``.

        The tensors ``x`` and ``y`` are expected to have shape ``(batch,...)``.
        If ``use_conv`` is enabled, the inputs are first flattened after a
        convolutional layer, otherwise the raw vectors are used.
        """
        if self.use_conv:
            x_f = self.conv(x).view(x.size(0), -1)
            y_f = self.conv(y).view(y.size(0), -1)
        else:
            x_f = x
            y_f = y
        diff = x_f - y_f
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: float = 1.0, use_conv: bool = False) -> np.ndarray:
    """Return the Gram matrix between two collections of tensors.

    Parameters
    ----------
    a, b : sequences of torch.Tensor
        Each element is a data point of shape ``(features,)`` or ``(1, H, W)``
        if ``use_conv`` is ``True``.
    gamma : float, default 1.0
        RBF width.
    use_conv : bool, default False
        Whether the convolutional pre‑processor is applied.

    Returns
    -------
    np.ndarray
        A 2‑D array ``K`` where ``K[i, j] = k(a[i], b[j])``.
    """
    kernel = HybridKernel(gamma=gamma, use_conv=use_conv)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridKernel", "kernel_matrix"]
