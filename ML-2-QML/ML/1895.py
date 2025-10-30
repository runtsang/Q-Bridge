"""Enhanced classical classifier with residual blocks, batch norm, and dropout.

The function build_classifier_circuit returns a torch.nn.Module that
consists of an input layer, a stack of residual blocks, and a final
linear head. It also returns metadata describing the encoding indices,
weight sizes per layer, and observable indices for a binary classifier.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """A residual block that preserves dimensionality."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.fc(x)) + x)


def build_classifier_circuit(
    num_features: int,
    depth: int,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Build a deep residual classifier for binary tasks.

    Parameters
    ----------
    num_features : int
        Number of input features / qubits.
    depth : int
        Number of residual blocks.

    Returns
    -------
    model : nn.Module
        The constructed classifier network.
    encoding : Iterable[int]
        Indices of input features (here simply 0..num_features-1).
    weight_sizes : Iterable[int]
        Number of trainable parameters per layer.
    observables : List[int]
        Indices of output neurons corresponding to class labels.
    """
    layers = [nn.Linear(num_features, num_features), nn.ReLU()]
    weight_sizes = [layers[0].weight.numel() + layers[0].bias.numel()]

    for _ in range(depth):
        block = ResidualBlock(num_features)
        layers.append(block)
        weight_sizes.append(
            block.fc.weight.numel()
            + block.fc.bias.numel()
            + block.bn.weight.numel()
            + block.bn.bias.numel()
        )

    head = nn.Linear(num_features, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    model = nn.Sequential(*layers)

    encoding = list(range(num_features))
    observables = [0, 1]  # simple binary classification

    return model, encoding, weight_sizes, observables
