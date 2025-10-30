"""Hybrid classical model with configurable head for classification, regression, or sampling."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple

def build_classifier_circuit(num_features: int, depth: int, mode: str = "classification") -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward network that mirrors the quantum ``build_classifier_circuit`` interface.
    Returned tuple: (model, encoding, weight_sizes, observables). ``encoding`` and ``observables`` are
    placeholders used by the quantum side.
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU(inplace=True)])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    # head depends on mode
    if mode == "classification":
        head = nn.Linear(in_dim, 2)
    elif mode == "regression":
        head = nn.Linear(in_dim, 1)
    elif mode == "sampler":
        head = nn.Linear(in_dim, 2)
    else:
        raise ValueError(f"unsupported mode {mode!r}")

    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2 if mode!= "regression" else 1))
    return network, encoding, weight_sizes, observables

class HybridModel(nn.Module):
    """
    A flexible PyTorch network that can act as a classifier, regressor or sampler.
    ``mode`` determines the architecture of the final layer and the loss function used.
    """
    def __init__(self, num_features: int, depth: int = 3, mode: str = "classification"):
        super().__init__()
        self.mode = mode
        self.features, _, _, _ = build_classifier_circuit(num_features, depth, mode)
        if mode == "sampler":
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        if self.mode == "sampler":
            out = self.softmax(out)
        return out

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == "classification":
            return F.cross_entropy(logits, targets.long())
        if self.mode == "regression":
            return F.mse_loss(logits.squeeze(-1), targets)
        if self.mode == "sampler":
            return F.nll_loss(torch.log(logits), targets.long())
        raise RuntimeError("unreachable")

__all__ = ["HybridModel", "build_classifier_circuit"]
