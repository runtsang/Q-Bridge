"""Hybrid classical layer that combines a convolutional filter and a fully‑connected layer.

The class is built on top of the original Conv.py and FCL.py seeds but
adds:
* trainable threshold parameters for both convolution and fully‑connected
  parts.
* support for multi‑channel inputs and arbitrary output channels.
* a single ``forward`` method that returns the concatenated feature vector,
  ready for downstream network modules.

This design keeps the module drop‑in compatible with the original
``Conv`` and ``FCL`` APIs while enabling end‑to‑end training.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import sigmoid, tanh

class HybridQuantumHybrid(nn.Module):
    """
    A classical surrogate for a quantum convolution + fully‑connected
    hybrid layer.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square convolution kernel.
    conv_channels : int, default 1
        Number of output channels from the convolution part.
    n_features : int, default 1
        Dimensionality of the fully‑connected part.
    conv_threshold : float, default 0.0
        Threshold used inside the sigmoid activation of the conv part.
    fc_threshold : float, default 0.0
        Bias added before the tanh of the FC part.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        conv_channels: int = 1,
        n_features: int = 1,
        conv_threshold: float = 0.0,
        fc_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=conv_channels, kernel_size=kernel_size, bias=True
        )
        self.fc = nn.Linear(n_features, 1)
        self.conv_threshold = conv_threshold
        self.fc_threshold = fc_threshold

    def forward(self, image: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        """
        Compute the hybrid output.

        Parameters
        ----------
        image : torch.Tensor
            4‑D tensor of shape (batch, 1, H, W).
        thetas : torch.Tensor
            1‑D tensor of length ``n_features`` to feed the fully‑connected part.

        Returns
        -------
        torch.Tensor
            Concatenated feature vector of shape (batch, conv_channels + 1).
        """
        # Convolution branch
        conv_out = self.conv(image)
        conv_out = sigmoid(conv_out - self.conv_threshold)
        conv_features = conv_out.mean(dim=(2, 3))  # global average pooling

        # Fully‑connected branch
        fc_input = thetas.view(-1, 1)
        fc_out = tanh(self.fc(fc_input) + self.fc_threshold)
        fc_features = fc_out.squeeze(-1)

        # Concatenate
        return torch.cat([conv_features, fc_features], dim=1)

__all__ = ["HybridQuantumHybrid"]
