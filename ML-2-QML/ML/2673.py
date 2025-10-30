"""Hybrid fully connected and quanvolution layer for classical training.

The class combines a simple convolutional front‑end inspired by the
``Quanvolution`` example with a linear fully connected head similar to
``FCL``.  It exposes a ``run`` method that mimics the quantum
parameter‑driven expectation value, but implemented purely with PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable


class HybridFCLQuanvolution(nn.Module):
    """
    Classical hybrid of a convolutional filter and a fully connected layer.

    Parameters
    ----------
    n_features : int, default 1
        Dimensionality of the output of the linear head.
    n_filters : int, default 4
        Number of convolutional filters.
    kernel_size : int, default 2
        Size of the convolutional kernel.
    stride : int, default 2
        Stride of the convolution.
    """

    def __init__(
        self,
        n_features: int = 1,
        n_filters: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, n_filters, kernel_size=kernel_size, stride=stride)
        # 28x28 input -> 14x14 feature map
        self.linear = nn.Linear(n_filters * 14 * 14, n_features)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Standard forward pass used for training."""
        features = self.conv(x)
        features = features.view(x.size(0), -1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Mimic a quantum expectation value.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameters that would normally drive a quantum circuit.
        Returns
        -------
        torch.Tensor
            A 1‑D tensor containing the mean activation of the linear head.
        """
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = self.activation(self.linear(theta_tensor))
        return out.mean(dim=0).detach().numpy()
