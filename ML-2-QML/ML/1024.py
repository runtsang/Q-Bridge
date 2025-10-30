"""Enhanced classical classifier mirroring the quantum helper interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """A simple residual block: Linear → ReLU → (Dropout) → Linear → + input."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = F.relu(out)
        return out + residual


class QuantumClassifierModel(nn.Module):
    """
    Classical feed‑forward network that mimics the quantum helper interface.
    Supports optional dropout and residual connections.
    """
    def __init__(
        self,
        num_features: int,
        depth: int = 3,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.dropout = dropout
        self.residual = residual

        layers: List[nn.Module] = []
        in_dim = num_features

        for _ in range(depth):
            if residual:
                block = ResidualBlock(in_dim, num_features, dropout)
                layers.append(block)
            else:
                linear = nn.Linear(in_dim, num_features)
                layers.append(linear)
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
            in_dim = num_features

        # Final head
        layers.append(nn.Linear(in_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def weight_sizes(self) -> List[int]:
        """Return the number of trainable parameters per linear layer."""
        sizes: List[int] = []
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                sizes.append(layer.weight.numel() + layer.bias.numel())
        return sizes


def build_classifier_circuit(
    num_features: int,
    depth: int,
    dropout: float = 0.0,
    residual: bool = False,
) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
    """
    Construct a feed‑forward classifier and metadata similar to the quantum variant.
    Returns:
        model: nn.Module
        encoding: Iterable[int] (feature indices)
        weight_sizes: List[int] (trainable params per layer)
        observables: List[int] (class indices)
    """
    model = QuantumClassifierModel(num_features, depth, dropout, residual)
    encoding = list(range(num_features))
    weight_sizes = model.weight_sizes()
    observables = list(range(2))
    return model, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
