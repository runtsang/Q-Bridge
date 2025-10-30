"""Hybrid fraud‑detection with learnable feature scaling and clipping regularisation.

The module builds a two‑layer neural network that mirrors the photonic architecture
and adds a per‑layer feature‑scale buffer and optional clipping.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, List

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
    feature_scale: float = 1.0
    bias: float = 0.0

class _Layer(nn.Module):
    def __init__(self, params: FraudLayerParameters, clip: bool):
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(torch.tensor(params.displacement_r, dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(params.displacement_phi, dtype=torch.float32))
        self.feature_scale = nn.Parameter(torch.tensor(params.feature_scale, dtype=torch.float32))
        self.bias_term = nn.Parameter(torch.tensor(params.bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        y = self.activation(self.linear(x))
        y = y * self.scale + self.shift
        y = y * self.feature_scale + self.bias_term
        return y

class FraudDetection(nn.Module):
    """Classical fraud‑detection model built from a list of layer parameters."""
    def __init__(self, input_params: FraudLayerParameters, hidden_params: Iterable[FraudLayerParameters], clip: bool = True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(_Layer(input_params, clip=False))
        for p in hidden_params:
            self.layers.append(_Layer(p, clip=clip))
        self.out = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.out(x)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    clip: bool = True
) -> nn.Sequential:
    """Instantiate a FraudDetection model and return its sequential representation."""
    model = FraudDetection(input_params, layers, clip=clip)
    # expose as nn.Sequential for compatibility with upstream code
    seq = nn.Sequential(*model.layers, model.out)
    return seq

__all__ = ["FraudLayerParameters", "FraudDetection", "build_fraud_detection_program"]
