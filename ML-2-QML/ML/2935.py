"""Hybrid classical regression model inspired by fraud‑detection layers.

This module builds a PyTorch model that mirrors the layered structure of the
photonic fraud‑detection example while adding a final linear head.  Each
layer is a custom ``nn.Module`` that applies a linear transformation,
a non‑linearity, and a learned affine scaling.  The first layer is
unclipped, subsequent layers are clipped to keep parameters within a
reasonable range, mirroring the photonic implementation.

The public API matches the original EstimatorQNN example: a factory
``EstimatorQNN`` that returns an ``EstimatorQNN__gen202`` instance.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Tuple

@dataclass
class FraudLayerParameters:
    """Parameters describing a single fraud‑layer block."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to [-bound, bound]."""
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
    """Create a single fraud‑layer module."""
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
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Construct a sequential model from a list of fraud‑layer parameters."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class EstimatorQNN__gen202(nn.Module):
    """Hybrid regressor that chains fraud‑layer blocks with a linear head."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_params: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.model = build_fraud_detection_program(input_params, hidden_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)

def EstimatorQNN(
    input_params: FraudLayerParameters,
    hidden_params: Iterable[FraudLayerParameters],
) -> EstimatorQNN__gen202:
    """Convenience factory mirroring the original API."""
    return EstimatorQNN__gen202(input_params, hidden_params)

__all__ = [
    "EstimatorQNN",
    "EstimatorQNN__gen202",
    "FraudLayerParameters",
    "build_fraud_detection_program",
]
