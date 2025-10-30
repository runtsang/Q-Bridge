"""Enhanced classical classifier factory mirroring the quantum helper interface."""
from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block with optional batch‑norm and dropout."""
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual
        return F.relu(out)


def build_classifier_circuit(num_features: int,
                             depth: int,
                             dropout: float = 0.0,
                             use_batchnorm: bool = True) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
    """Construct a feed‑forward residual classifier.

    Parameters
    ----------
    num_features: int
        Dimensionality of the input feature vector.
    depth: int
        Number of residual blocks to stack.
    dropout: float, optional
        Dropout probability applied after each residual block.
    use_batchnorm: bool, optional
        Whether to include batch‑norm layers inside the blocks.

    Returns
    -------
    network: nn.Module
        The assembled PyTorch model.
    encoding: Iterable[int]
        Indices of features that are directly fed into the network.
    weight_sizes: List[int]
        Number of trainable parameters per layer (useful for profiling).
    observables: List[int]
        Dummy observable indices (class labels) to keep API parity.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    # Initial linear mapping
    layers.append(nn.Linear(in_dim, num_features))
    layers.append(nn.ReLU())

    weight_sizes: List[int] = [layers[0].weight.numel() + layers[0].bias.numel()]

    # Residual blocks
    for _ in range(depth):
        block = ResidualBlock(num_features, dropout=dropout)
        layers.append(block)
        # Each block has two Linear layers
        for linear in [block.fc1, block.fc2]:
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())

    # Final classification head
    head = nn.Linear(num_features, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    observables = [0, 1]  # placeholder for class indices
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
