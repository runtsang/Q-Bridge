"""Hybrid classifier module for the classical side.

Provides a feed‑forward network that mirrors the quantum interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn


@dataclass
class LayerParameters:
    weight: torch.Tensor
    bias: torch.Tensor
    scale: torch.Tensor
    shift: torch.Tensor


def _clip_tensor(t: torch.Tensor, bound: float) -> torch.Tensor:
    return t.clamp(-bound, bound)


def _layer_from_params(params: LayerParameters, *, clip: bool = True) -> nn.Module:
    weight = params.weight.clone()
    bias = params.bias.clone()
    if clip:
        weight = _clip_tensor(weight, 5.0)
        bias = _clip_tensor(bias, 5.0)

    linear = nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = nn.Tanh()
            self.register_buffer("scale", params.scale)
            self.register_buffer("shift", params.shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(x))
            out = out * self.scale + self.shift
            return out

    return Layer()


class HybridClassifier:
    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        *,
        clip: bool = True,
    ) -> Tuple[nn.Sequential, Iterable[int], Iterable[int], list[int]]:
        """Construct a feed‑forward classifier that mimics the quantum helper."""
        layers: list[nn.Module] = []
        weight_sizes: list[int] = []

        # Input layer
        weight = torch.randn(num_features, num_features)
        bias = torch.randn(num_features)
        scale = torch.ones(num_features)
        shift = torch.zeros(num_features)
        layers.append(_layer_from_params(LayerParameters(weight, bias, scale, shift), clip=clip))
        weight_sizes.append(weight.numel() + bias.numel())

        # Hidden layers
        for _ in range(depth - 1):
            weight = torch.randn(num_features, num_features)
            bias = torch.randn(num_features)
            scale = torch.ones(num_features)
            shift = torch.zeros(num_features)
            layers.append(_layer_from_params(LayerParameters(weight, bias, scale, shift), clip=clip))
            weight_sizes.append(weight.numel() + bias.numel())

        # Output layer
        weight = torch.randn(2, num_features)
        bias = torch.randn(2)
        scale = torch.ones(2)
        shift = torch.zeros(2)
        layers.append(_layer_from_params(LayerParameters(weight, bias, scale, shift), clip=clip))
        weight_sizes.append(weight.numel() + bias.numel())

        network = nn.Sequential(*layers)

        encoding = list(range(num_features))
        observables = [0, 1]
        return network, encoding, weight_sizes, observables


__all__ = ["HybridClassifier", "LayerParameters"]
