"""Hybrid convolutional filter with classical RBF kernel.

This module provides a drop-in replacement for the original Conv filter,
combining a standard 2‑D convolution with a radial‑basis‑function kernel
to compute a similarity score between the input patch and a learned kernel
weight.  The design supports scaling across large classical datasets while
remaining lightweight for deployment.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class HybridConvKernel(nn.Module):
    """
    Hybrid convolutional kernel that fuses a 2‑D convolution with an RBF
    similarity measure.  The class is intentionally lightweight so it can
    serve as a drop‑in replacement for the legacy Conv filter in
    ``Conv.py``.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        gamma: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size: int
            Size of the square convolution kernel.
        threshold: float
            Bias threshold applied before the sigmoid activation.
        gamma: float
            RBF kernel width (inverse squared length scale).
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.gamma = gamma

        # Classical convolution layer
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )

    def forward(self, data: np.ndarray) -> float:
        """
        Compute a similarity score for a single ``kernel_size``×``kernel_size``
        patch.  The patch is first passed through a convolution followed by
        a sigmoid activation; the resulting feature map is then compared to
        the convolution weights using an RBF kernel.

        Parameters
        ----------
        data: np.ndarray
            2‑D array of shape ``(kernel_size, kernel_size)``.

        Returns
        -------
        float
            RBF similarity between the convolved patch and the kernel weights.
        """
        # Convert to torch tensor and reshape to match Conv2d input
        tensor = torch.as_tensor(data, dtype=torch.float32).view(
            1, 1, self.kernel_size, self.kernel_size
        )
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)

        # Flatten both the activations and the kernel weights
        act_vec = activations.view(-1)
        weight_vec = self.conv.weight.view(-1)

        # Compute RBF kernel value
        diff = act_vec - weight_vec
        rbf = torch.exp(-self.gamma * torch.sum(diff * diff))
        return rbf.item()


__all__ = ["HybridConvKernel"]
