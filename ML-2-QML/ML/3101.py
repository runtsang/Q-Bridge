"""Hybrid fraud detection model combining photonic‑style layers and feature encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    num_features: int = 2,
    encoding_depth: int = 1,
) -> nn.Sequential:
    """
    Construct a hybrid model that first encodes raw features into a 2‑D space,
    then applies a stack of photonic‑style layers, and finally a linear head.
    """
    # Feature encoder: linear mapping to 2‑dim space followed by activation
    encoder = nn.Sequential(
        nn.Linear(num_features, 2),
        nn.ReLU(),
    )

    # Photonic‑style body
    body_layers: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    body_layers.extend(_layer_from_params(layer, clip=True) for layer in layers)

    # Classifier head
    head = nn.Linear(2, 1)

    return nn.Sequential(encoder, *body_layers, head)


def weight_summary(network: nn.Module) -> List[int]:
    """Return a list of the number of trainable parameters in each sub‑module."""
    return [p.numel() for p in network.parameters() if p.requires_grad]


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "weight_summary",
]
