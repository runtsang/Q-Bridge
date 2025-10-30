"""Hybrid classical model combining fraud detection layers and a quantum-inspired classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
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
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Sequential, List[int], List[int], List[int]]:
    """Build a hybrid classifier: fraud detection layers followed by a quantum-inspired feed‑forward network."""
    # Fraud‑detection portion (fixed simple example)
    input_params = FraudLayerParameters(
        bs_theta=0.6,
        bs_phi=0.4,
        phases=(0.05, -0.05),
        squeeze_r=(0.1, 0.1),
        squeeze_phi=(0.0, 0.0),
        displacement_r=(1.0, 1.0),
        displacement_phi=(0.0, 0.0),
        kerr=(0.0, 0.0),
    )
    fraud_layers = [
        FraudLayerParameters(
            bs_theta=0.5,
            bs_phi=0.3,
            phases=(0.1, -0.1),
            squeeze_r=(0.2, 0.2),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
    ]
    fraud_net = build_fraud_detection_program(input_params, fraud_layers)

    # Quantum‑inspired feed‑forward network
    layers: List[nn.Module] = []
    in_dim = 2  # fraud_net output dimension
    encoding = list(range(in_dim))
    weight_sizes: List[int] = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    classifier = nn.Sequential(*layers)
    full_net = nn.Sequential(fraud_net, classifier)
    observables = list(range(2))
    return full_net, encoding, weight_sizes, observables


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "build_classifier_circuit"]
