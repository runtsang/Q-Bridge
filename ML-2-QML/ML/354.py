"""Enhanced classical classifier with residual blocks and feature‑wise dropout."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn

def build_classifier_circuit(
    num_features: int,
    depth: int,
    num_classes: int = 2,
    dropout_rate: float = 0.1,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Build a residual‑style feed‑forward network that mirrors the quantum interface.

    Parameters
    ----------
    num_features : int
        Number of input features for the original training data.
    depth : int
        Number of residual blocks.
    num_classes : int, default 2
        Number of output classes.
    dropout_rate : float, default 0.1
        Dropout probability applied to feature dimension between blocks.

    Returns
    -------
    model : nn.Module
        Residual network.
    encoding : Iterable[int]
        Placeholder for compatibility with quantum interface (feature indices).
    weight_sizes : Iterable[int]
        Number of trainable parameters per linear layer.
    observables : List[int]
        List of class indices, used as placeholder for observables.
    """
    class ResidualBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.relu(self.linear(x))

    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    # Initial linear layer
    linear = nn.Linear(in_dim, num_features)
    layers.append(linear)
    weight_sizes.append(linear.weight.numel() + linear.bias.numel())
    layers.append(nn.ReLU())

    # Residual blocks
    for _ in range(depth):
        block = ResidualBlock(num_features)
        layers.append(block)
        weight_sizes.append(block.linear.weight.numel() + block.linear.bias.numel())
        layers.append(nn.Dropout(dropout_rate))

    # Final classifier head
    head = nn.Linear(num_features, num_classes)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    model = nn.Sequential(*layers)
    observables = list(range(num_classes))
    return model, encoding, weight_sizes, observables
