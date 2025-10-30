"""Hybrid classical estimator that mirrors the quantum EstimatorQNN.

The model first applies a 2‑D convolutional filter (kernel size 2 by
default) to the input data, then feeds the resulting scalar into a
tiny fully‑connected regression network.  This design reproduces the
behaviour of the quantum estimator while remaining entirely classical
and PyTorch‑compatible.
"""

from __future__ import annotations

import torch
from torch import nn


class EstimatorQNNHybrid(nn.Module):
    """Hybrid estimator combining a classical convolution and a
    feed‑forward regression network.

    Attributes
    ----------
    conv : nn.Conv2d
        2‑D convolutional filter that emulates the quantum filter.
    regressor : nn.Sequential
        Tiny fully‑connected network that performs regression on the
        scalar feature produced by the convolution.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Classical convolutional filter
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )

        # Small regression network
        self.regressor = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (..., H, W) or (batch, 1, H, W).  The
            tensor is first padded to the required kernel size, convolved,
            and the mean activation is taken as a scalar feature.

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch, 1).
        """
        # Ensure 4‑D shape: (batch, 1, H, W)
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0).unsqueeze(0)
        elif inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)

        # Convolution
        logits = self.conv(inputs)
        activations = torch.sigmoid(logits - self.threshold)
        feature = activations.mean(dim=(2, 3), keepdim=True)

        # Regression
        return self.regressor(feature)


__all__ = ["EstimatorQNNHybrid"]
