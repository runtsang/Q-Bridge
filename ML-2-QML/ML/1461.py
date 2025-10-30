"""Enhanced classical fraud detection model with dropout, batchnorm, and configurable activations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    activation: str = "tanh"  # default activation


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to a symmetric interval."""
    return max(-bound, min(bound, value))


class _ScaleShift(nn.Module):
    """Scale and shift module to emulate photonic displacement."""

    def __init__(self, scale: torch.Tensor, shift: torch.Tensor):
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x * self.scale + self.shift


def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
    dropout: float,
    use_batchnorm: bool,
) -> nn.Module:
    """Construct a single layer with optional regularisation."""
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi], [params.squeeze_r[0], params.squeeze_r[1]]],
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

    activation_cls = getattr(nn, params.activation.capitalize())
    activation = activation_cls()

    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    modules = [linear, activation]
    if use_batchnorm:
        modules.append(nn.BatchNorm1d(2))
    if dropout > 0.0:
        modules.append(nn.Dropout(dropout))
    modules.append(_ScaleShift(scale, shift))

    return nn.Sequential(*modules)


class FraudDetector(nn.Module):
    """Classical fraud detection model with configurable depth and regularisation."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.model = self._build_model(input_params, layers, dropout, use_batchnorm)

    @staticmethod
    def _build_model(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout: float,
        use_batchnorm: bool,
    ) -> nn.Sequential:
        modules = [
            _layer_from_params(
                input_params, clip=False, dropout=dropout, use_batchnorm=use_batchnorm
            )
        ]
        modules.extend(
            _layer_from_params(
                layer, clip=True, dropout=dropout, use_batchnorm=use_batchnorm
            )
            for layer in layers
        )
        modules.append(nn.Linear(2, 1))
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *,
    dropout: float = 0.0,
    use_batchnorm: bool = False,
) -> nn.Sequential:
    """Expose a function compatible with the original API."""
    return FraudDetector(
        input_params, layers, dropout=dropout, use_batchnorm=use_batchnorm
    ).model


__all__ = ["FraudLayerParameters", "FraudDetector", "build_fraud_detection_program"]
