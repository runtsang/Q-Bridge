"""Hybrid classical classifier that emulates a quantum‑style feature extractor.

The network consists of a 2×2 stride‑2 convolution (mimicking a quantum patch
kernel), a flattening layer, and a linear head.  Metadata about the
encoding indices, weight sizes, and output observables is provided to
facilitate a direct comparison with the quantum implementation.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Build a classical feed‑forward network with metadata that mirrors a
    quantum circuit.  The network uses a shallow fully‑connected stack
    followed by a binary head.

    Parameters
    ----------
    num_features : int
        Dimensionality of each input feature vector.
    depth : int
        Number of hidden layers.

    Returns
    -------
    network : nn.Module
        The constructed classifier.
    encoding : Iterable[int]
        Indices of features that would be encoded in a quantum circuit.
    weight_sizes : Iterable[int]
        Number of trainable parameters per linear layer.
    observables : List[int]
        Dummy observable indices for compatibility with the QML side.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

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
    observables = [0, 1]  # placeholder for binary classification
    return network, encoding, weight_sizes, observables


class HybridClassifier(nn.Module):
    """
    Classical hybrid model that replaces a quantum patch‑wise filter with a
    lightweight 2×2 convolution.  The architecture is deliberately
    lightweight to allow rapid prototyping while still providing a
    comparable number of parameters to the quantum circuit.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        # 2×2 kernel, stride 2: 28×28 → 14×14 feature map
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return F.log_softmax(logits, dim=-1)


__all__ = ["build_classifier_circuit", "HybridClassifier"]
