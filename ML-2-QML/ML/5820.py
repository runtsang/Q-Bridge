"""QCNNGen216 – Classical implementation inspired by QCNN and QuantumClassifierModel."""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple, List


def _build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Construct a feed‑forward classifier with *depth* hidden layers.
    Returns:
        network: nn.Sequential classifier
        encoding: indices of input features (mirrors quantum encoding)
        weight_sizes: number of trainable parameters per linear layer
        observables: dummy observable indices (2‑class output)
    """
    layers: List[nn.Module] = []
    weight_sizes: List[int] = []
    in_dim = num_features

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
    encoding = list(range(num_features))
    return network, encoding, weight_sizes, observables


class QCNNGen216(nn.Module):
    """
    Classical convolutional network that emulates the QCNN structure.
    The network consists of a feature map, a stack of conv/pool pairs,
    and a depth‑controlled classifier head.
    """

    def __init__(self, num_features: int = 8, depth: int = 3) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth

        # Feature map: linear projection + non‑linearity
        self.feature_map = nn.Sequential(nn.Linear(num_features, 16), nn.Tanh())

        # Convolution + pooling blocks
        self.convs: nn.ModuleList = nn.ModuleList()
        self.pools: nn.ModuleList = nn.ModuleList()
        in_dim = 16
        for _ in range(depth):
            conv = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Tanh())
            pool = nn.Sequential(nn.Linear(in_dim, max(1, in_dim // 2)), nn.Tanh())
            self.convs.append(conv)
            self.pools.append(pool)
            in_dim = pool[-1].out_features

        # Classifier head
        self.classifier, self.encoding, self.weight_sizes, self.observables = _build_classifier_circuit(in_dim, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        for conv, pool in zip(self.convs, self.pools):
            x = conv(x)
            x = pool(x)
        logits = self.classifier(x)
        return torch.sigmoid(logits)

    def get_metadata(self) -> Tuple[List[int], List[int], List[int]]:
        """Return encoding, weight sizes, and observables for comparison."""
        return self.encoding, self.weight_sizes, self.observables


__all__ = ["QCNNGen216"]
