"""Enhanced classical classifier with residual connections and configurable dropout.

Provides a factory that mirrors the quantum interface, enabling seamless comparison in experiments.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn


class QuantumClassifierModel:
    """Factory for a residual neural network classifier.

    The API closely matches the quantum helper:
    - ``build(num_features, depth, hidden_size=None, dropout=0.0)`` returns
      ``(network, encoding, weight_sizes, observables)``.
    """

    class ResidualBlock(nn.Module):
        """Single residual block with optional projection."""

        def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self.proj = (
                nn.Identity()
                if in_dim == out_dim
                else nn.Linear(in_dim, out_dim, bias=False)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = self.proj(x)
            out = self.linear(x)
            out = self.relu(out)
            out = self.dropout(out)
            return out + residual

    @staticmethod
    def build(
        num_features: int,
        depth: int,
        hidden_size: Optional[int] = None,
        dropout: float = 0.0,
    ) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
        """
        Parameters
        ----------
        num_features : int
            Number of input features.
        depth : int
            Number of hidden layers.
        hidden_size : int | None, default None
            Size of each hidden layer. If None, uses ``num_features``.
        dropout : float, default 0.0
            Dropout probability applied after each hidden ReLU.

        Returns
        -------
        network : nn.Module
            Residual neural network.
        encoding : Iterable[int]
            Indices of input features used at the first layer.
        weight_sizes : List[int]
            Number of trainable parameters per block (including projection).
        observables : List[int]
            Indices of the output logits (here always ``[0, 1]``).
        """
        hidden_size = hidden_size or num_features

        layers: List[nn.Module] = []
        weight_sizes: List[int] = []

        in_dim = num_features
        for _ in range(depth):
            block = QuantumClassifierModel.ResidualBlock(in_dim, hidden_size, dropout)
            layers.append(block)
            # Count parameters: linear + projection (if any)
            weight_sizes.append(
                block.linear.weight.numel() + block.linear.bias.numel()
            )
            if not isinstance(block.proj, nn.Identity):
                weight_sizes.append(block.proj.weight.numel())
            in_dim = hidden_size

        # Output head
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        encoding = list(range(num_features))
        observables = [0, 1]  # two-class logits
        return network, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel"]
