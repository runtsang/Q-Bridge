"""Enhanced classical classifier mirroring quantum interface with residual blocks."""
from __future__ import annotations
from typing import Iterable, Tuple
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Simple residual block with two linear layers."""
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.linear1(x))
        out = self.linear2(out)
        return self.act(out + residual)

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Construct a residual feed‑forward classifier with metadata matching the quantum helper."""
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []

    # initial linear layer
    layers.append(nn.Linear(in_dim, num_features))
    weight_sizes.append(layers[-1].weight.numel() + layers[-1].bias.numel())

    # residual blocks
    for _ in range(depth):
        block = ResidualBlock(num_features)
        layers.append(block)
        weight_sizes.append(
            block.linear1.weight.numel() + block.linear1.bias.numel() +
            block.linear2.weight.numel() + block.linear2.bias.numel()
        )

    # final classifier head
    head = nn.Linear(num_features, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]  # placeholder indices for binary classification
    return network, encoding, weight_sizes, observables

__all__ = ["build_classifier_circuit"]
