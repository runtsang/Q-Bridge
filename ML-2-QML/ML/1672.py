"""Enhanced classical classifier with residual connections and regularisation."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


def build_classifier_circuit(
    num_features: int,
    depth: int,
    *,
    dropout_prob: float = 0.25,
    use_batchnorm: bool = True,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward neural network that mirrors the quantum helper while
    providing advanced regularisation and residual connectivity.

    Parameters
    ----------
    num_features : int
        Dimensionality of each input vector.
    depth : int
        Number of hidden residual blocks.
    dropout_prob : float, optional
        Drop‑out probability applied after each block.
    use_batchnorm : bool, optional
        Whether to include a batch‑normalisation layer after the linear
        transformation.

    Returns
    -------
    Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]
        * A PyTorch `nn.Sequential` model.
        * The indices of the input features (identity mapping).
        * A list containing the number of trainable parameters per block.
        * A list of observable indices (here simply `[0, 1]` for two‑class output).
    """
    class ResidualBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.bn = nn.BatchNorm1d(dim) if use_batchnorm else nn.Identity()
            self.dropout = nn.Dropout(dropout_prob)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x) + x

    layers = []
    weight_sizes = []

    for _ in range(depth):
        block = ResidualBlock(num_features)
        layers.append(block)
        layers.append(nn.ReLU())
        layers.append(block.dropout)
        weight_sizes.append(block.linear.weight.numel() + block.linear.bias.numel())

    head = nn.Linear(num_features, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    observables = [0, 1]
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
