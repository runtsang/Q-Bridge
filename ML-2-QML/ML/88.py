"""Enhanced fraud detection model with modular layers and optional regularization.

The model mirrors the original photonic structure but augments it with
dropout, batch‑normalization and a flexible number of hidden layers.
It can be used directly with PyTorch optimizers and loss functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
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


class FraudDetectionModel(nn.Module):
    """Modular fraud‑detection network with optional regularization."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_params: Iterable[FraudLayerParameters],
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
        for params in hidden_params:
            layers.append(_layer_from_params(params, clip=True))
            if batch_norm:
                layers.append(nn.BatchNorm1d(2))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(2, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    return FraudDetectionModel(
        input_params=input_params,
        hidden_params=layers,
        dropout=0.1,
        batch_norm=True,
    )


__all__ = ["FraudLayerParameters", "FraudDetectionModel", "build_fraud_detection_program"]
