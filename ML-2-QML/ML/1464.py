"""Classical gated multi‑class classifier mirroring the quantum interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedLinear(nn.Module):
    """Linear layer with a feature‑wise sigmoid gate."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gate   = nn.Linear(in_features, out_features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * torch.sigmoid(self.gate(x))

def build_classifier_circuit(num_features: int, depth: int, num_classes: int = 3
                             ) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
    """
    Construct a gated feed‑forward classifier and metadata that mirrors the quantum helper.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input.
    depth : int
        Number of gated hidden layers.
    num_classes : int, default=3
        Number of target classes.

    Returns
    -------
    network : nn.Module
        Sequential model with gated layers and a final linear head.
    encoding : Iterable[int]
        Indices of the input feature columns (used for compatibility with the QML side).
    weight_sizes : List[int]
        Number of learnable parameters per linear (including bias) in the order they appear.
    observables : List[int]
        Dummy observables list matching the quantum side; here just the class indices.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        gated = GatedLinear(in_dim, num_features)
        layers.extend([gated, nn.ReLU()])
        weight_sizes.append(
            gated.linear.weight.numel() + gated.linear.bias.numel()
            + gated.gate.weight.numel() + gated.gate.bias.numel()
        )
        in_dim = num_features

    head = nn.Linear(in_dim, num_classes)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(num_classes))
    return network, encoding, weight_sizes, observables

__all__ = ["build_classifier_circuit"]
