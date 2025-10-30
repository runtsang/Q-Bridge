"""Hybrid fraud detection engine – classical implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass
class FraudDetectionParams:
    """Parameters for a single neural layer and global hyper‑parameters."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    dropout: float = 0.0
    clip: bool = False
    weight_clip: float | None = None
    bias_clip: float | None = None


def _clip(value: torch.Tensor, bound: float) -> torch.Tensor:
    """Clamp tensor values to a symmetric interval."""
    return torch.clamp(value, -bound, bound)


def _layer_from_params(params: FraudDetectionParams, *, clip: bool) -> nn.Module:
    """Construct an individual layer with optional clipping and dropout."""
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)

    if clip:
        if params.weight_clip is not None:
            weight = _clip(weight, params.weight_clip)
        if params.bias_clip is not None:
            bias = _clip(bias, params.bias_clip)

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
            self.dropout = nn.Dropout(params.dropout) if params.dropout > 0.0 else nn.Identity()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.linear(inputs)
            x = self.activation(x)
            x = x * self.scale + self.shift
            x = self.dropout(x)
            return x

    return Layer()


def build_fraud_detection_program(
    input_params: FraudDetectionParams,
    layers: Iterable[FraudDetectionParams],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudDetectionEngine:
    """Encapsulates a classical fraud‑detection neural network."""

    def __init__(self, input_params: FraudDetectionParams, layers: Sequence[FraudDetectionParams]) -> None:
        self.model = build_fraud_detection_program(input_params, layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


__all__ = ["FraudDetectionParams", "build_fraud_detection_program", "FraudDetectionEngine"]
