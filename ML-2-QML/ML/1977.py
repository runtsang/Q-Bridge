"""Enhanced classical neural classifier mirroring the quantum helper interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel(nn.Module):
    """
    A flexible feed‑forward network that mirrors the quantum classifier API.
    Parameters
    ----------
    num_features : int
        Size of the input feature vector.
    hidden_dims : Sequence[int]
        Number of units per hidden layer.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    batch_norm : bool, optional
        Whether to include a batch‑normalization layer after each hidden layer.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: List[int] | Tuple[int,...] = (64, 32),
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = out_dim
        # Classification head
        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)

    @staticmethod
    def build_classifier_circuit(
        num_features: int, depth: int
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Return a lightweight representation of the classifier for compatibility
        with the quantum‑side API. The returned tuple contains:
        - the network,
        - a list of indices representing the encoding positions,
        - a list of weight‑parameter counts per layer,
        - a list of observable indices (here simply `[0, 1]`).
        """
        # Build a simple feed‑forward net with one hidden layer of size `num_features`
        layers = nn.ModuleList()
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        net = nn.Sequential(*layers)
        encoding = list(range(num_features))
        weight_sizes = [l.weight.numel() + l.bias.numel() for l in layers if isinstance(l, nn.Linear)]
        observables = [0, 1]
        return net, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel"]
