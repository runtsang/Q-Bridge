"""Hybrid classical classifier that can include a fraud‑detection style head."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn


@dataclass
class FraudLayerParams:
    """Parameters for a fraud‑detection style fully connected layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParams, clip: bool = False) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParams,
    layers: Iterable[FraudLayerParams],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


def _build_classifier_network(
    num_features: int,
    depth: int,
    fraud_layer: Optional[FraudLayerParams] = None,
) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    # Base feed‑forward part
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    # Optional fraud‑detection inspired head
    if fraud_layer is not None:
        layers.append(build_fraud_detection_program(fraud_layer, []))
        # Rough placeholder: actual size depends on params
        weight_sizes.append(4)

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    net = nn.Sequential(*layers)
    observables = list(range(2))
    return net, encoding, weight_sizes, observables


class HybridClassifier(nn.Module):
    """Classical hybrid classifier that can include a fraud‑detection style head."""
    def __init__(
        self,
        num_features: int,
        depth: int = 1,
        fraud_layer: Optional[FraudLayerParams] = None,
    ) -> None:
        super().__init__()
        self.model, self.encoding, self.weight_sizes, self.observables = _build_classifier_network(
            num_features, depth, fraud_layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_metadata(self) -> Tuple[List[int], List[int], List[int]]:
        """Return encoding, weight sizes and observables for benchmarking."""
        return self.encoding, self.weight_sizes, self.observables


__all__ = ["HybridClassifier", "FraudLayerParams"]
