from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import torch
from torch import nn
import numpy as np

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

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(inputs))
            return out * self.scale + self.shift

    return Layer()

class FCL(nn.Module):
    """Classical fully‑connected layer mimicking the quantum FCL example."""
    def __init__(self, n_features: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x)).mean(dim=0, keepdim=True)

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud‑detection model combining photonic‑inspired classical layers
    with a fully‑connected quantum‑inspired layer."""
    def __init__(self, fraud_params: Iterable[FraudLayerParameters], n_features: int = 2):
        super().__init__()
        modules = []
        for i, param in enumerate(fraud_params):
            modules.append(_layer_from_params(param, clip=(i > 0)))
        self.classical = nn.Sequential(*modules)
        self.fcl = FCL(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.classical(x)
        out = self.fcl(out)
        return out

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid", "generate_superposition_data", "FCL"]
