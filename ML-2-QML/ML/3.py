"""Hybrid classical classifier with optional residual and batch‑norm layers.

The builder mirrors the quantum circuit signature and returns a feed‑forward
network together with metadata that can be consumed by a hybrid training
pipeline.  The network optionally inserts residual connections (identity
skip) between layers and can prepend a BatchNorm1d after each linear
transformation.

The API is intentionally identical to the quantum helper so that the
model can be swapped in a one‑liner.
"""
from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A single residual block with optional batch‑norm."""
    def __init__(self, features: int, batch_norm: bool = False):
        super().__init__()
        self.linear = nn.Linear(features, features)
        self.bn = nn.BatchNorm1d(features) if batch_norm else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.bn is not None:
            out = self.bn(out)
        out = self.relu(out)
        return out + x


def build_classifier_circuit(
    num_features: int,
    depth: int,
    use_residual: bool = False,
    batch_norm: bool = False,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward network that mirrors the quantum circuit’s
    structure, but with optional residual connections and batch‑norm.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers.
    use_residual : bool, default False
        If True, each hidden layer is wrapped in a ResidualBlock.
    batch_norm : bool, default False
        If True, a BatchNorm1d layer is inserted after each linear layer.

    Returns
    -------
    network : nn.Module
        The assembled network (nn.Sequential).
    encoding : List[int]
        List of indices that represent the input feature positions.
    weight_sizes : List[int]
        Number of trainable parameters per block (including the head).
    observables : List[int]
        Dummy observable list for compatibility with quantum helper.
    """
    layers: List[nn.Module] = []
    weight_sizes: List[int] = []

    in_dim = num_features
    for _ in range(depth):
        if use_residual:
            block = ResidualBlock(in_dim, batch_norm=batch_norm)
            layers.append(block)
            weight_sizes.append(
                block.linear.weight.numel() + block.linear.bias.numel()
            )
        else:
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            if batch_norm:
                layers.append(nn.BatchNorm1d(num_features))
            layers.append(nn.ReLU())
        in_dim = num_features

    # Classification head
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    observables = list(range(2))  # placeholder for compatibility
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
