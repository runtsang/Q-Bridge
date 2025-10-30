"""Hybrid classical convolutional filter and classifier.

This module merges the classical convolution filter from Conv.py
with the feed‑forward classifier from QuantumClassifierModel.py.
The resulting :class:`HybridConvClassifier` is a pure PyTorch
module that can be used as a drop‑in replacement for a quantum
convolutional neural network.

Features
--------
* 2×2 convolution filter with optional thresholding.
* Multi‑layer ReLU classifier.

Usage
-----
>>> model = HybridConvClassifier(kernel_size=2, depth=3)
>>> logits = model(torch.rand(1, 1, 2, 2))
>>> probs = torch.softmax(logits, dim=1)
"""
from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple


def build_classifier(num_features: int, depth: int) -> Tuple[nn.Sequential, Iterable[int], list[int]]:
    """
    Construct a feed‑forward classifier and return the network,
    a list of input feature indices and the list of parameter
    counts per layer.

    Parameters
    ----------
    num_features : int
        Number of input features.
    depth : int
        Number of hidden layers.

    Returns
    -------
    network : nn.Sequential
        The classifier.
    encoding : list[int]
        Indices of features that are used as input (identity here).
    weight_sizes : list[int]
        Number of trainable parameters per layer.
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    return network, encoding, weight_sizes


class HybridConvClassifier(nn.Module):
    """
    Classical hybrid convolution + classifier.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the 2‑D convolution filter.
    threshold : float, default 0.0
        Threshold applied before the sigmoid activation.
    depth : int, default 3
        Depth of the feed‑forward classifier.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Convolution filter
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )

        # Feed‑forward classifier
        num_features = 1  # filter produces a single scalar feature
        self.classifier, self.encoding, self.weight_sizes = build_classifier(
            num_features=num_features,
            depth=depth,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W).

        Returns
        -------
        logits : torch.Tensor
            Raw classifier scores of shape (B, 2).
        """
        # Apply convolution filter
        conv_out = self.conv(x)
        # Reshape to (B, 1)
        conv_out = conv_out.view(x.size(0), -1)
        # Sigmoid activation with threshold
        conv_out = torch.sigmoid(conv_out - self.threshold)

        # Classifier expects features of shape (B, num_features)
        logits = self.classifier(conv_out)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper that returns class probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W).

        Returns
        -------
        probs : torch.Tensor
            Probabilities of shape (B, 2).
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


__all__ = ["HybridConvClassifier"]
