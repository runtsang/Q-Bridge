"""Hybrid classical classifier combining feed‑forward architecture with fraud‑detection inspired custom layers.

The model accepts a sequence of FraudLayerParameters that encode linear weights, biases, scaling and shift.
It exposes a static method ``build_classifier_circuit`` that mirrors the quantum helper interface
and returns the network, encoding indices, weight sizes and output observables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
import torch.nn as nn

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
    # Build weight matrix from bs_theta, bs_phi and squeeze parameters
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
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class HybridClassifier(nn.Module):
    """Feed‑forward classifier that mirrors the quantum interface.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers.
    layer_params : Iterable[FraudLayerParameters] | None
        Optional explicit layer parameters; if omitted a random initialization is used.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        layer_params: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.layers: nn.ModuleList = nn.ModuleList()

        # Build hidden layers
        for i in range(depth):
            params = next(iter(layer_params)) if layer_params else FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(1.0, 1.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
            self.layers.append(_layer_from_params(params, clip=True))

        # Final classification head
        self.head = nn.Linear(num_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return self.head(out)

    @staticmethod
    def build_classifier_circuit(
        num_features: int, depth: int
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """Construct a hybrid network and return metadata for quantum‑classical comparison."""
        layers = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes = []

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

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "HybridClassifier"]
