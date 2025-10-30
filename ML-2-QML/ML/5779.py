"""Classic binary classifier that mirrors the quantum architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """
    Construct a feed‑forward architecture with the same metadata as the quantum ansatz.

    Parameters
    ----------
    num_features : int
        Number of input features.
    depth : int
        Number of hidden blocks.

    Returns
    -------
    network : nn.Sequential
        Sequential network consisting of `depth` blocks of Linear → ReLU.
    encoding : list[int]
        Indices of input features used for encoding (identity mapping).
    weight_sizes : list[int]
        Number of trainable parameters per linear layer.
    observables : list[int]
        Dummy observable indices matching the quantum implementation.
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

class HybridBinaryClassifierML(nn.Module):
    """
    Classical binary classifier that uses the same layer depth and feature mapping
    as the quantum version.  The final logits are converted to probabilities
    via a softmax.  This class is fully differentiable and can be trained
    with any standard PyTorch optimiser.
    """
    def __init__(self, num_features: int, depth: int = 3) -> None:
        super().__init__()
        self.network, self.encoding, self.weight_sizes, _ = build_classifier_circuit(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        return F.softmax(logits, dim=-1)

__all__ = ["build_classifier_circuit", "HybridBinaryClassifierML"]
