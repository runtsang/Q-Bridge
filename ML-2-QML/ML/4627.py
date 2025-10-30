"""Hybrid quantum‑classical classifier that mirrors the quantum interface while staying purely classical.

The module provides:
* build_classifier_circuit – a feed‑forward network with metadata for a quantum counterpart.
* build_fraud_detection_program – a sequence of layers inspired by photonic fraud‑detection circuits.
* EstimatorQNN – a tiny regressor that can be used as a hybrid backend.
* HybridClassifier – a convenient nn.Module that bundles the network and its metadata.

This design allows easy comparison between classical and quantum embeddings and facilitates experimentation
with hybrid training pipelines.
"""

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


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a feed‑forward classifier and metadata similar to the quantum variant."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

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


def EstimatorQNN() -> nn.Module:
    """Return a tiny fully‑connected regression network."""
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(inputs)

    return EstimatorNN()


class HybridClassifier(nn.Module):
    """Convenience wrapper that exposes both the network and its metadata."""
    def __init__(self, num_features: int, depth: int) -> None:
        super().__init__()
        self.net, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "build_classifier_circuit",
    "EstimatorQNN",
    "HybridClassifier",
]
