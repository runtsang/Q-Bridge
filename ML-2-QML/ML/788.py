"""Extended classical fraud‑detection model with dropout and batch‑norm."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import torch
from torch import nn


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


class FraudDetectionModel(nn.Module):
    """Hybrid classical fraud detection model with optional dropout and batch‑norm."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_params: Iterable[FraudLayerParameters],
        *,
        dropout_prob: float = 0.0,
        batch_norm: bool = False,
        clip_weights: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
        for params in hidden_params:
            layer = _layer_from_params(params, clip=clip_weights)
            layers.append(layer)
            if batch_norm:
                layers.append(nn.BatchNorm1d(2))
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(p=dropout_prob))
        layers.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_fraud_detection_model(
    input_params: FraudLayerParameters,
    hidden_params: Iterable[FraudLayerParameters],
    *,
    dropout_prob: float = 0.0,
    batch_norm: bool = False,
    clip_weights: bool = True,
) -> FraudDetectionModel:
    """Convenience constructor mirroring the original API."""
    return FraudDetectionModel(
        input_params,
        hidden_params,
        dropout_prob=dropout_prob,
        batch_norm=batch_norm,
        clip_weights=clip_weights,
    )


__all__ = [
    "FraudLayerParameters",
    "FraudDetectionModel",
    "build_fraud_detection_model",
]
