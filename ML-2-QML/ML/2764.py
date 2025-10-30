"""Hybrid classifier combining classical neural network layers with fraud‑detection style parameters.

This module preserves the interface of the original *QuantumClassifierModel* while
injecting the parameter‑clipping and custom activation logic from the fraud‑detection
example.  The resulting `HybridClassifier` exposes a `forward` method that can be
used in standard PyTorch pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn


@dataclass
class FraudLayerParameters:
    """Parameters that describe a single fraud‑detection style layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a value to ``[-bound, bound]``."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Translate a `FraudLayerParameters` instance into a PyTorch layer."""
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

    linear = nn.Linear(2, 2, bias=True)
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
    input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]
) -> nn.Sequential:
    """Create a sequential PyTorch model that mimics the photonic fraud‑detection stack."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


def build_classifier_circuit(
    num_features: int, depth: int, fraud_params: Iterable[FraudLayerParameters]
) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Construct a hybrid classifier that combines a fraud‑detection head with a feed‑forward stack.

    Parameters
    ----------
    num_features : int
        Number of input features (must be 2 to match the fraud layers).
    depth : int
        Depth of the fully‑connected backbone.
    fraud_params : iterable
        Sequence of `FraudLayerParameters` used for the fraud detection head.
    """
    # Fraud detection head
    fraud_head = build_fraud_detection_program(fraud_params[0], fraud_params[1:])

    # Feed‑forward backbone
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)

    backbone = nn.Sequential(*layers)

    # Combine fraud head and backbone into a single module
    class HybridClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fraud_head = fraud_head
            self.backbone = backbone
            self.final = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            fraud_output = self.fraud_head(x[:, :2])
            backbone_output = self.backbone(x)
            combined = torch.cat([fraud_output, backbone_output], dim=-1)
            return self.final(combined)

    network = HybridClassifier()
    weight_sizes = [p.numel() for p in network.parameters()]
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


__all__ = ["HybridClassifier", "FraudLayerParameters", "build_classifier_circuit"]
