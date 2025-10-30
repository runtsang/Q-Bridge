from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterable

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

class ScaleShift(nn.Module):
    """Applies element‑wise scaling and shifting to a 2‑dimensional tensor."""
    def __init__(self, scale: torch.Tensor, shift: torch.Tensor) -> None:
        super().__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift

class FraudDetection(nn.Module):
    """A PyTorch model that emulates the layered photonic circuit."""
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.input_layer = self._make_layer(input_params, clip=False)
        self.hidden_layers = nn.ModuleList([self._make_layer(p, clip=True) for p in layers])
        self.output = nn.Linear(2, 1)

    def _make_layer(self, params: FraudLayerParameters, clip: bool) -> nn.Module:
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
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
        return nn.Sequential(linear, activation, ScaleShift(scale, shift))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output(x)
