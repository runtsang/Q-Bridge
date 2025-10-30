"""Enhanced classical classifier with embedding and residual blocks."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


def build_classifier_circuit(
    num_features: int,
    depth: int,
    embedding_dim: int = 32,
    skip_first: bool = True,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward network that augments the seed with an
    embedding layer, optional residual connections, and a flexible
    skip‑first flag.

    Parameters
    ----------
    num_features : int
        Dimensionality of the raw input feature vector.
    depth : int
        Number of hidden residual blocks.
    embedding_dim : int, default 32
        Size of the learned embedding that projects the raw input
        into a richer space.
    skip_first : bool, default True
        Whether to add a skip connection from the embedding output
        to each hidden block.

    Returns
    -------
    nn.Module
        Fully‑connected network ready for training.
    Iterable[int]
        List of input feature indices (identity mapping).
    Iterable[int]
        List of weight‑parameter counts for each layer.
    list[int]
        Observable indices (here the two output logits).
    """
    layers: list[nn.Module] = []

    # Embedding and projection to match hidden‑layer dimensionality.
    embedding = nn.Linear(num_features, embedding_dim)
    projection = nn.Linear(embedding_dim, num_features)
    layers.append(embedding)
    layers.append(nn.ReLU())
    layers.append(projection)
    layers.append(nn.ReLU())

    weight_sizes: list[int] = []
    weight_sizes.append(embedding.weight.numel() + embedding.bias.numel())
    weight_sizes.append(projection.weight.numel() + projection.bias.numel())

    # Residual block definition.
    class ResidualBlock(nn.Module):
        def __init__(self, linear: nn.Linear):
            super().__init__()
            self.linear = linear
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(self.linear(x) + x)

    # Build hidden residual blocks.
    for _ in range(depth):
        linear = nn.Linear(num_features, num_features)
        layers.append(ResidualBlock(linear))
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())

    # Final classification head.
    head = nn.Linear(num_features, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)

    encoding = list(range(num_features))
    observables = list(range(2))

    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
