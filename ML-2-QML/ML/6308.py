"""Hybrid classical classifier mirroring the quantum interface.

The module defines a quanvolutional feature extractor followed by a
feed‑forward head.  The returned metadata (`encoding`, `weight_sizes`,
`observables`) is compatible with the quantum counterpart, enabling
joint or transfer learning between the two representations.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Two‑pixel patch convolution followed by flattening."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)

class HybridClassifier(nn.Module):
    """Classical hybrid model: quanvolution filter → linear head."""
    def __init__(self, num_features: int = 64, hidden_dim: int = 128,
                 num_classes: int = 2) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # The filter outputs `num_features` features
        self.linear = nn.Linear(num_features, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        h = F.relu(self.linear(features))
        logits = self.head(h)
        return F.log_softmax(logits, dim=-1)

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module,
                                                                   Iterable[int],
                                                                   Iterable[int],
                                                                   List[int]]:
    """
    Construct a classical network that mimics the quantum interface.

    Parameters
    ----------
    num_features : int
        Number of features produced by the quanvolution filter.
    depth : int
        Depth of the feed‑forward head (number of hidden layers).

    Returns
    -------
    network : nn.Module
        Feed‑forward network with `depth` hidden layers.
    encoding : Iterable[int]
        Indices of the input features (0 … num_features‑1).
    weight_sizes : Iterable[int]
        Number of trainable parameters per layer.
    observables : List[int]
        Dummy observable indices (class labels).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
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
    encoding = list(range(num_features))
    observables = [0, 1]  # class indices

    return network, encoding, weight_sizes, observables

__all__ = ["QuanvolutionFilter", "HybridClassifier", "build_classifier_circuit"]
