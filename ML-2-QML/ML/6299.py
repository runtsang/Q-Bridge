"""HybridClassifier: a clip‑aware, depth‑parameterised classical feed‑forward model.

This module mirrors the quantum helper interface while adding the ability to
clip weights and biases, inspired by the fraud‑detection seed.  The public
API is identical to the original `build_classifier_circuit` and can be used
interchangeably with the quantum counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


@dataclass
class LayerParams:
    """Container for a single linear layer's parameters."""
    weight: torch.Tensor
    bias: torch.Tensor


def _clip(tensor: torch.Tensor, bound: float) -> torch.Tensor:
    """Clamp tensor values to the interval [-bound, bound]."""
    return tensor.clamp(-bound, bound)


def _layer_from_params(params: LayerParams, *, clip: bool) -> nn.Module:
    """Create a linear + activation + optional clipping layer."""
    linear = nn.Linear(params.weight.shape[1], params.weight.shape[0])
    with torch.no_grad():
        linear.weight.copy_(params.weight)
        linear.bias.copy_(params.bias)
    if clip:
        linear.weight.data = _clip(linear.weight.data, 5.0)
        linear.bias.data = _clip(linear.bias.data, 5.0)
    activation = nn.Tanh()
    return nn.Sequential(linear, activation)


def build_classifier_circuit(
    num_features: int,
    depth: int,
    *,
    clip: bool = False,
) -> Tuple[nn.Sequential, Iterable[int], List[int], List[int]]:
    """
    Construct a feed‑forward classifier with optional weight clipping.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input and hidden layers.
    depth : int
        Number of hidden layers.
    clip : bool, optional
        If True, clip all weights and biases to [-5, 5] after construction.

    Returns
    -------
    network : nn.Sequential
        The assembled model.
    encoding : Iterable[int]
        Indices of input features (identity mapping).
    weight_sizes : List[int]
        Number of trainable parameters per layer, useful for bookkeeping.
    observables : List[int]
        Dummy observable identifiers; retained for API compatibility.
    """
    layers: List[nn.Module] = []
    weight_sizes: List[int] = []

    # Input layer
    in_dim = num_features
    for _ in range(depth):
        weight = torch.randn(num_features, in_dim)
        bias = torch.randn(num_features)
        layer = _layer_from_params(LayerParams(weight, bias), clip=clip)
        layers.append(layer)
        weight_sizes.append(weight.numel() + bias.numel())
        in_dim = num_features

    # Output head
    head = nn.Linear(in_dim, 2)
    if clip:
        head.weight.data = _clip(head.weight.data, 5.0)
        head.bias.data = _clip(head.bias.data, 5.0)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    observables = [0, 1]  # placeholder for two-class output
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit", "LayerParams"]
