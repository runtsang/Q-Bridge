"""Hybrid classical model with optional convolutional filtering and data‑upload style encoding.

The model can be used for classification or regression.  It contains:
* an optional 2‑D convolutional filter that mimics the quantum quanvolution
* a data‑upload encoder that generates a feature vector of sin/cos components
* a depth‑separable feed‑forward network
* a linear head that outputs class logits or a regression value
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Tuple


# --------------------------------------------------------------------------- #
#  2‑D convolutional filter (classical emulation of quanvolution)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Return a callable object that emulates the quantum filter with PyTorch ops."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply the filter to a batch of images.

        Parameters
        ----------
        data : torch.Tensor
            Batch of images with shape (batch, 1, H, W) where H, W >= kernel_size.

        Returns
        -------
        torch.Tensor
            Filtered activations of shape (batch, 1, H-kernel_size+1, W-kernel_size+1).
        """
        return torch.sigmoid(self.conv(data) - self.threshold)


# --------------------------------------------------------------------------- #
#  Data‑upload style encoder (sin/cos representation)
# --------------------------------------------------------------------------- #
def encode_features(x: torch.Tensor) -> torch.Tensor:
    """
    Encode real‑valued features via sine/cosine mapping.

    Parameters
    ----------
    x : torch.Tensor
        Input features of shape (..., num_features).

    Returns
    -------
    torch.Tensor
        Encoded features of shape (..., 2 * num_features).
    """
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


# --------------------------------------------------------------------------- #
#  Hybrid model definition
# --------------------------------------------------------------------------- #
class HybridModel(nn.Module):
    """
    Hybrid classical model that optionally incorporates a convolutional filter
    and a data‑upload style encoder.

    Parameters
    ----------
    num_features : int
        Number of input features (excluding image channels if ``use_conv`` is True).
    depth : int, default 3
        Number of hidden layers.
    use_conv : bool, default False
        If True, apply a 2‑D convolutional filter to image inputs.
    kernel_size : int, default 2
        Size of the convolutional kernel.
    threshold : float, default 0.0
        Threshold used by the sigmoid activation in the conv filter.
    output_dim : int, default 2
        Size of the output layer (2 for binary classification, 1 for regression).
    regression : bool, default False
        If True, the model is configured for regression.
    """

    def __init__(
        self,
        num_features: int,
        depth: int = 3,
        use_conv: bool = False,
        kernel_size: int = 2,
        threshold: float = 0.0,
        output_dim: int = 2,
        regression: bool = False,
    ) -> None:
        super().__init__()
        self.use_conv = use_conv
        self.regression = regression
        self.num_features = num_features
        self.depth = depth

        # Convolutional front‑end (optional)
        if self.use_conv:
            self.conv = ConvFilter(kernel_size=kernel_size, threshold=threshold)
            conv_out = (kernel_size // kernel_size) * 2 * num_features  # simplified
        else:
            conv_out = 2 * num_features  # encode_features doubles dimensionality

        # Depth‑separable feed‑forward network
        layers: list[nn.Module] = []
        in_dim = conv_out
        hidden_dim = max(conv_out // 2, 16)
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            If ``use_conv`` is True, ``x`` should have shape
            (batch, 1, H, W) where H, W >= kernel_size.
            Otherwise, ``x`` should have shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            Logits (classification) or regression value.
        """
        # optional convolution
        if self.use_conv:
            x = self.conv.run(x)
            # flatten spatial dims
            x = x.view(x.size(0), -1)
        else:
            x = encode_features(x)

        return self.net(x)


# --------------------------------------------------------------------------- #
#  Utility: build a classical classifier circuit (metadata only)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward classifier and metadata similar to the quantum variant.

    Returns
    -------
    Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]
        (network, encoding indices, weight sizes per layer, observable indices)
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables
