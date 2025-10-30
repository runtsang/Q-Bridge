"""Hybrid classical model combining classification and regression tasks."""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable, Tuple

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Construct a shared encoder for multi‑task learning.

    The encoder is a stack of `depth` linear/ReLU blocks. Two separate heads
    are attached: a 2‑logit classification head and a single‑output regression
    head.  The function returns the full module together with the indices of
    trainable parameters and a lightweight observable list for API parity
    with the quantum implementation.
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    encoder = nn.Sequential(*layers)
    cls_head = nn.Linear(in_dim, 2)
    reg_head = nn.Linear(in_dim, 1)
    weight_sizes.extend([cls_head.weight.numel() + cls_head.bias.numel(),
                         reg_head.weight.numel() + reg_head.bias.numel()])
    network = nn.ModuleDict({"encoder": encoder, "cls_head": cls_head, "reg_head": reg_head})
    observables = list(range(2))  # placeholder for compatibility
    return network, encoding, weight_sizes, observables

class HybridQuantumClassifierRegressor(nn.Module):
    """Classical counterpart to the quantum hybrid model.

    The forward method accepts a batch of feature vectors and returns a
    tuple of classification logits and regression predictions.
    """
    def __init__(self, num_features: int, depth: int = 3):
        super().__init__()
        self.network, _, _, _ = build_classifier_circuit(num_features, depth)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder = self.network["encoder"]
        cls_head = self.network["cls_head"]
        reg_head = self.network["reg_head"]
        features = encoder(x)
        return cls_head(features), reg_head(features).squeeze(-1)

__all__ = ["build_classifier_circuit", "HybridQuantumClassifierRegressor"]
