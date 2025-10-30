"""Hybrid classical fully‑connected layer built on a convolutional front‑end.

The module is designed to be a drop‑in replacement for the original
`FCL` seed while adding spatial feature extraction.  It mirrors the
quantum interface by exposing a `run` method that accepts a list of
tunable angles and returns a NumPy array of the mean activation.

The scaling paradigm is *combination*: a classical convolution kernel
is fused with a fully‑connected linear layer, keeping the overall
model lightweight and fully differentiable with PyTorch.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
from torch import nn
import numpy as np


class HybridFCL(nn.Module):
    """
    Classical hybrid layer that first applies a 2‑D convolution to the
    input data and then projects the flattened feature map through a
    linear transform.  The `run` method accepts a list of angles that
    modulate the linear layer weights via a simple tanh non‑linearity.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        n_features: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the square convolution kernel.
        conv_threshold : float
            Threshold used in the sigmoid activation after the conv.
        n_features : int
            Number of output features of the linear layer.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=kernel_size, bias=True
        )
        # Linear layer expects flattened conv output
        self.linear = nn.Linear(kernel_size * kernel_size, n_features)

    def run(self, thetas: Iterable[float], data: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Forward pass that applies the convolution, a sigmoid activation,
        then a linear layer whose weights are modulated by `thetas`.

        Parameters
        ----------
        thetas : Iterable[float]
            List of tunable angles; one per linear output feature.
        data : Sequence[Sequence[float]]
            2‑D array representing a single image patch.

        Returns
        -------
        np.ndarray
            Mean activation value of the linear layer after tanh.
        """
        # Convert data to torch tensor
        tensor = torch.as_tensor(data, dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.conv_threshold)
        # Flatten and apply linear layer
        flat = activations.view(1, -1)
        # Modulate linear weights by thetas
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(1, -1)
        # Simple weight modulation: element‑wise multiplication
        weight = self.linear.weight * theta_tensor
        bias = self.linear.bias
        output = torch.nn.functional.linear(flat, weight, bias)
        # Return mean of tanh activation
        expectation = torch.tanh(output).mean(dim=0)
        return expectation.detach().cpu().numpy()


def FCL() -> HybridFCL:
    """Return a fully‑connected hybrid layer mimicking the original API."""
    return HybridFCL()


__all__ = ["FCL"]
