"""Hybrid convolutional layer with a fully connected read‑out."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np


class HybridConvLayer(nn.Module):
    """
    Classical implementation of a convolutional filter followed by a fully
    connected layer.  The class is intentionally lightweight so it can be
    dropped in place of a quantum quanvolution layer.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel.
    threshold : float, default 0.0
        Bias applied before the sigmoid activation.
    fc_features : int, default 1
        Output dimensionality of the fully connected read‑out.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, fc_features: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        # Convolution with a single input and output channel
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        # Fully connected read‑out
        self.fc = nn.Linear(kernel_size * kernel_size, fc_features)

    def run(self, data: np.ndarray) -> float:
        """
        Apply the convolution, activation, flatten, and fully connected layer.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) representing a patch.

        Returns
        -------
        float
            Scalar output of the layer.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size)
        conv_out = self.conv(tensor)
        activated = torch.sigmoid(conv_out - self.threshold)
        flat = activated.view(-1)
        fc_out = self.fc(flat)
        # Reduce to a single scalar for compatibility with the quantum API
        return fc_out.mean().item()


__all__ = ["HybridConvLayer"]
