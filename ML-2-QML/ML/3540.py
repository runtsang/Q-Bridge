"""Hybrid classical estimator combining quanvolution and deep regression.

The module defines:
* ``QuanvolutionFilter`` – a 2×2 patch extractor implemented with a
  single 2‑stride convolution.
* ``EstimatorQNN__gen166`` – a neural net that first applies the
  quanvolution filter, then passes the flattened features through
  several hidden layers with Tanh activations, finally producing a
  single regression output.

The architecture mirrors the original EstimatorQNN example but
augments it with a feature extraction stage inspired by the
Quanvolution example.  All weights are Torch tensors and the network
is fully differentiable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """
    Classical 2×2 patch extractor.

    The filter reduces a 28×28 image to a 14×14 grid of 4‑dimensional
    features.  It is implemented as a single convolutional layer with
    kernel size 2 and stride 2; the output is flattened per sample.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Feature vector of shape (batch, 4 * 14 * 14).
        """
        features = self.conv(x)
        return features.view(x.size(0), -1)


class EstimatorQNN__gen166(nn.Module):
    """
    Hybrid estimator that combines a quanvolution filter with a deep
    feed‑forward network.

    The network is structured as:
        quanvolution → linear → tanh → linear → tanh → linear → output
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.fc1 = nn.Linear(4 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Predicted scalar value per sample.
        """
        features = self.qfilter(x)
        out = self.fc1(features)
        out = torch.tanh(out)
        out = self.fc2(out)
        out = torch.tanh(out)
        out = self.fc3(out)
        return out


__all__ = ["QuanvolutionFilter", "EstimatorQNN__gen166"]
