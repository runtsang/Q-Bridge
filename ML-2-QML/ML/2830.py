"""Hybrid fraud‑detection model with quantum‑inspired feature mapping.

This module defines a PyTorch model that mirrors the photonic architecture
from the original seed but augments it with a purely classical layer that
mimics a quantum expectation value.  The model can be used independently
from any quantum backend, making it suitable for rapid prototyping
and unit testing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn
import numpy as np


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
    """Simple clipping helper used by the original photonic implementation."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single linear + Tanh + affine layer from the given parameters."""
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


class QuantumFeatureMapLayer(nn.Module):
    """A lightweight, quantum‑inspired feature map implemented with sinusoids.

    The layer accepts a 2‑D tensor and returns a 2‑D tensor that simulates a
    quantum expectation value of a parameterised gate.  It is deliberately
    pure PyTorch so that the model remains fully classical.
    """
    def __init__(self, n_features: int = 2, n_params: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_params, n_features))
        self.bias = nn.Parameter(torch.randn(n_params))
        self.linear = nn.Linear(n_params, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # simulate a quantum expectation: sin(weight * x + bias)
        z = torch.sin(torch.matmul(x, self.weight.t()) + self.bias)
        return self.linear(z)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure.

    The first classical layer is built without clipping, subsequent layers
    are clipped to keep parameters bounded.  A quantum‑inspired feature map
    is inserted before the final linear output.
    """
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(QuantumFeatureMapLayer())
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "QuantumFeatureMapLayer"]
