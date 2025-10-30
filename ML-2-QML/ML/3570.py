"""
ConvGen198 – Classical hybrid layer integrating a convolutional filter
and a fully‑connected block.

The class is drop‑in compatible with the original Conv.py anchor and
extends it to include a small linear head.  The design follows the
structure from the Conv and FCL seeds while adding a clear scaling
strategy: a single channel convolution followed by a linear output layer.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvGen198(nn.Module):
    """
    A hybrid classical layer that performs a kernel‑size 2 convolution
    with a thresholded sigmoid activation and then maps the result to a
    single output through a linear + tanh mapping.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        fc_features: int = 1,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        # Convolutional filter (1 input channel → 1 output channel)
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )
        self.threshold = threshold
        # Fully‑connected head
        self.linear = nn.Linear(1, fc_features)
        self.activation = nn.Tanh()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        data : torch.Tensor
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Scalar output after the fully‑connected head.
        """
        # Ensure correct shape: (batch, channel, H, W)
        tensor = torch.as_tensor(data, dtype=torch.float32).view(1, 1, -1, -1)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        # Flatten and feed to linear layer
        flat = activations.view(-1)
        linear_out = self.linear(flat.unsqueeze(0))
        output = self.activation(linear_out).mean()
        return output


__all__ = ["ConvGen198"]
