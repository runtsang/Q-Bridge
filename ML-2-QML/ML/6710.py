"""Quantum-inspired classical classifier with residual layers."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block with two linear layers and ReLU."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.lin1(x))
        out = self.lin2(out)
        return F.relu(out + residual)


class QuantumClassifierModel(nn.Module):
    """
    Classical classifier that mirrors the interface of the quantum helper.
    It uses a stack of residual blocks followed by a linear head.
    """

    def __init__(self, num_features: int, depth: int, hidden_dim: int | None = None):
        """
        Args:
            num_features: dimensionality of the input feature vector.
            depth: number of residual blocks.
            hidden_dim: width of the hidden layers; defaults to ``num_features``.
        """
        super().__init__()
        hidden_dim = hidden_dim or num_features
        self.blocks: nn.ModuleList = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(depth)]
        )
        self.head = nn.Linear(hidden_dim, 2)

    @staticmethod
    def build_classifier_circuit(
        num_features: int, depth: int, hidden_dim: int | None = None
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Construct a feed‑forward residual classifier and return
        metadata that mimics the quantum variant.

        Returns:
            network: the constructed :class:`~torch.nn.Module`.
            encoding: indices of input features (used as placeholders in the quantum API).
            weight_sizes: list of the number of trainable parameters in each layer.
            observables: list of target class indices (0 and 1).
        """
        hidden_dim = hidden_dim or num_features
        network = QuantumClassifierModel(num_features, depth, hidden_dim)
        encoding = list(range(num_features))
        weight_sizes = [p.numel() for p in network.parameters()]
        observables = [0, 1]
        return network, encoding, weight_sizes, observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)

__all__ = ["QuantumClassifierModel"]
