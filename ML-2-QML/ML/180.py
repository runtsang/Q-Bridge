"""Enhanced classical classifier with depth‑wise feature‑wise kernels and stochastic dropout."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Iterable, Tuple, List

class DepthwiseLinear(nn.Module):
    """Per‑feature linear layer (diagonal weight matrix)."""
    def __init__(self, num_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features))
        self.bias = nn.Parameter(torch.randn(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier with
    - depth‑wise feature‑wise kernels,
    - stochastic dropout after each ReLU,
    - and a final linear head.

    Returns:
        network: nn.Sequential model
        encoding: list of feature indices (identity encoding)
        weight_sizes: list of number of trainable parameters per layer
        observables: list of output classes (two‑class classification)
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.2))
        depthwise = DepthwiseLinear(num_features)
        layers.append(depthwise)

        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        weight_sizes.append(depthwise.weight.numel() + depthwise.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables
