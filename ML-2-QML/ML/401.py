"""Hybrid classical classifier with residual blocks and dropout for richer feature extraction."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward classifier with optional residual connections and dropout.
    Returns:
        network: nn.Sequential model
        encoding: indices of input features (for compatibility with quantum interface)
        weight_sizes: list of number of parameters per layer
        observables: list of output class indices (here 0 and 1)
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []

    def residual_block(in_dim: int, out_dim: int) -> nn.Module:
        """Single residual block with two linear layers and dropout."""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    for i in range(depth):
        if i % 2 == 0:
            block = residual_block(in_dim, num_features)
            layers.append(block)
            weight_sizes.extend([p.numel() for p in block.parameters()])
            in_dim = num_features
        else:
            linear = nn.Linear(in_dim, num_features)
            layers.append(nn.Sequential(linear, nn.ReLU(), nn.Dropout(0.3)))
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]
    return network, encoding, weight_sizes, observables

__all__ = ["build_classifier_circuit"]
