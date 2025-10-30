"""Hybrid classical convolution + feed‑forward estimator.

This module implements a drop‑in replacement for the quantum‑convolution
layer in a purely classical setting.  The design merges the
convolutional feature extraction of the original `Conv` seed with the
fully‑connected regression of `EstimatorQNN`.  The resulting
`HybridConvEstimator` can be trained with standard PyTorch optimisers
and is ready for integration into larger CNN pipelines.

The class exposes:
* `conv` – a 2‑D convolution with learnable bias.
* `regressor` – a small MLP that maps the flattened convolutional
  activations to a single regression output.
* `forward` – standard PyTorch forward pass.
* `run` – a convenience method that mimics the original API and
  returns a scalar value.

The implementation intentionally favours a clean, single‑module
structure and avoids any quantum dependencies.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple


class HybridConvEstimator(nn.Module):
    """
    Classical hybrid of a convolutional filter and a lightweight regressor.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        regressor_hidden: Tuple[int,...] = (8, 4),
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the square convolutional kernel.
        threshold : float
            Bias value added to the convolution before the sigmoid.
        regressor_hidden : tuple
            Sequence of hidden layer widths for the MLP.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Convolutional feature extractor
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )

        # Simple MLP for regression
        layers = []
        input_dim = 1 * kernel_size * kernel_size
        for hidden in regressor_hidden:
            layers += [nn.Linear(input_dim, hidden), nn.Tanh()]
            input_dim = hidden
        layers.append(nn.Linear(input_dim, 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass through the convolution and the regressor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W) where H=W=kernel_size.

        Returns
        -------
        torch.Tensor
            Batch of scalar predictions.
        """
        # Convolution -> sigmoid -> flatten
        conv_out = torch.sigmoid(self.conv(x) - self.threshold)
        flat = conv_out.view(conv_out.size(0), -1)
        return self.regressor(flat)

    def run(self, data: torch.Tensor) -> float:
        """
        Convenience method that replicates the original `Conv.run` API.
        It expects a single 2‑D array of shape (kernel_size, kernel_size).

        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            The mean activation after the sigmoid and the regressor output.
        """
        if data.dim() == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
        with torch.no_grad():
            out = self.forward(data)
        return out.item()


__all__ = ["HybridConvEstimator"]
