"""Hybrid classical kernel combining RBF with a 2‑D convolutional filter."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class HybridKernel(nn.Module):
    """Classical RBF kernel augmented with a convolutional feature extractor.

    Parameters
    ----------
    gamma : float, default=1.0
        RBF width parameter.
    conv_kernel_size : int, default=2
        Size of the square convolutional kernel.
    conv_threshold : float, default=0.0
        Activation threshold for the sigmoid output of the convolutional filter.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.conv_kernel_size = conv_kernel_size
        self.conv_threshold = conv_threshold

        # Convolutional filter: single‑channel, single‑output.
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=conv_kernel_size,
            bias=True,
        )
        # Slightly non‑trivial initialisation to break symmetry.
        nn.init.constant_(self.conv.weight, 0.1)
        nn.init.constant_(self.conv.bias, 0.0)

    def _conv_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the 2‑D convolutional filter and return a scalar feature.

        The input vector ``x`` must have length ``conv_kernel_size**2``.
        """
        size = self.conv_kernel_size
        if x.numel()!= size * size:
            raise ValueError(
                f"Input length must be {size * size} for a {size}×{size} filter."
            )
        tensor = x.view(1, 1, size, size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.conv_threshold)
        return activations.mean()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the hybrid RBF kernel between two 1‑D tensors.

        Each vector is augmented with its convolutional feature before
        the RBF computation, giving the kernel a richer representation.
        """
        f_x = self._conv_feature(x)
        f_y = self._conv_feature(y)
        # Augment the feature vectors with the scalar convolution output.
        x_aug = torch.cat([x, f_x.unsqueeze(0)])
        y_aug = torch.cat([y, f_y.unsqueeze(0)])
        diff = x_aug - y_aug
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Return the Gram matrix for two collections of vectors."""
        return np.array(
            [[self(x, y).item() for y in b] for x in a]
        )


__all__ = ["HybridKernel"]
