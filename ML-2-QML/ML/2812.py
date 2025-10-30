"""Hybrid QCNN model combining classical convolution‑inspired layers and a quantum‑inspired classifier head.

This module defines:
* `QCNNHybrid` – a torch.nn.Module that mirrors the QCNN ansatz with configurable depth.
* `build_classifier_circuit` – a helper that creates a feed‑forward classifier with the same metadata as the quantum variant, facilitating side‑by‑side experiments.
"""

from __future__ import annotations

from typing import Tuple, List

import torch
from torch import nn


class QCNNHybrid(nn.Module):
    """A classical QCNN‑style network.

    The architecture follows the same convolution–pooling pattern as the quantum ansatz:
    a feature map, a stack of convolutional blocks (linear + ReLU), optional pooling
    layers, and a linear classifier head.  The depth of the stack is a hyper‑parameter,
    allowing a direct comparison with the quantum counterpart that uses an identical
    layer count.
    """
    def __init__(self, input_dim: int = 8, depth: int = 3, hidden_dim: int = 16) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

        # Convolutional and pooling blocks
        self.conv_layers: nn.ModuleList = nn.ModuleList()
        self.pool_layers: nn.ModuleList = nn.ModuleList()
        for _ in range(depth):
            self.conv_layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
            self.pool_layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        # Classifier head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = pool(x)
        return torch.sigmoid(self.head(x))


def QCNNHybrid() -> QCNNHybrid:
    """Factory that returns a QCNNHybrid instance with default hyper‑parameters."""
    return QCNNHybrid()


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Sequential, List[int], List[int], List[int]]:
    """Construct a feed‑forward classifier and metadata mirroring the quantum helper.

    The returned tuple consists of:
    * the network (nn.Sequential)
    * a list of encoding indices (simply the feature indices)
    * the number of trainable parameters per layer
    * a list of observable indices (classical analogue of quantum observables)
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

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


__all__ = ["QCNNHybrid", "build_classifier_circuit"]
