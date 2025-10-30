"""Hybrid classical convolution module integrating RBF kernels and a QCNN‑style FC stack."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from torch import nn


class HybridConv(nn.Module):
    """
    Classical hybrid convolution model.

    The forward pass performs:
    1. 2‑D convolution with a learnable kernel (default size 2).
    2. Thresholded sigmoid activation and mean pooling.
    3. Feature extraction via a radial basis function (RBF) kernel.
    4. A QCNN‑inspired fully‑connected stack ending in a scalar output.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the convolution kernel.  Default is 2.
    threshold : float, optional
        Threshold applied before the sigmoid.  Default is 0.0.
    rbf_gamma : float, optional
        Gamma parameter for the RBF kernel.  Default is 1.0.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        rbf_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.rbf_gamma = rbf_gamma

        # 1‑D convolution on a single‑channel image
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )

        # QCNN‑style FC stack
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (B, 1, H, W) or (H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, 1) with values in (0, 1).
        """
        if inputs.dim() == 2:
            # Add batch and channel dimensions
            inputs = inputs.unsqueeze(0).unsqueeze(0)

        # 1. Convolution
        conv_out = self.conv(inputs)  # shape: (B, 1, H', W')

        # 2. Thresholded sigmoid and mean
        logits = conv_out - self.threshold
        activated = torch.sigmoid(logits)
        pooled = activated.mean(dim=(2, 3), keepdim=True)  # shape: (B, 1, 1, 1)

        # 3. Flatten for kernel feature extraction
        features = pooled.view(pooled.size(0), -1)  # shape: (B, 1)

        # 4. RBF kernel embedding (optional, here we just keep the scalar)
        #    In practice, one could concatenate the kernel vector to the FC stack.
        #    For simplicity, we proceed directly to the FC stack.
        #    The RBF kernel is exposed as a static helper below.

        # 5. QCNN‑style FC stack
        x = self.feature_map(features)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        logits = self.head(x)
        return torch.sigmoid(logits)

    @staticmethod
    def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        """
        Compute the RBF kernel between two tensors.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of the same shape.
        gamma : float, optional
            Kernel parameter.  Default is 1.0.

        Returns
        -------
        torch.Tensor
            Scalar kernel value.
        """
        diff = x - y
        return torch.exp(-gamma * torch.sum(diff * diff))

    @staticmethod
    def kernel_matrix(
        a: Iterable[torch.Tensor], b: Iterable[torch.Tensor], gamma: float = 1.0
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of tensors using the RBF kernel.

        Parameters
        ----------
        a, b : iterable of torch.Tensor
            Collections of 1‑D tensors.
        gamma : float, optional
            Kernel parameter.

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        return np.array(
            [[HybridConv.rbf_kernel(x, y, gamma).item() for y in b] for x in a]
        )


def Conv() -> HybridConv:
    """
    Factory function that returns a drop‑in replacement for the original `Conv` filter.
    """
    return HybridConv()


__all__ = ["HybridConv", "Conv", "HybridConv.kernel_matrix"]
