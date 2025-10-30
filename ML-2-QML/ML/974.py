"""Classical fraud detection model using PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

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
    depth: int = 1  # future‑proofing for deeper architectures


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
    activation: nn.Module,
    dropout: float,
) -> nn.Module:
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

    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.dropout = nn.Dropout(dropout)
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.activation(self.linear(inputs))
            x = self.dropout(x)
            x = x * self.scale + self.shift
            return x

    return Layer()


class FraudDetectionModel(nn.Module):
    """Hybrid classical fraud detection model with configurable layers."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = [
            _layer_from_params(input_params, clip=False, activation=activation, dropout=dropout)
        ]
        modules.extend(
            _layer_from_params(layer, clip=True, activation=activation, dropout=dropout)
            for layer in layers
        )
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @classmethod
    def build_from_params(
        cls,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.1,
    ) -> "FraudDetectionModel":
        return cls(input_params, layers, activation=activation, dropout=dropout)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
