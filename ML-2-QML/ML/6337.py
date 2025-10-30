"""Enhanced classical fraud‑detection pipeline with residual connections and feature‑wise scaling."""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = ["FraudLayerParameters", "FraudFeatureScaler", "ResidualFraudLayer", "FraudDetectionEnhanced"]


@dataclass
class FraudLayerParameters:
    """Parameters for a single FraudLayer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudFeatureScaler(nn.Module):
    """Feature‑wise linear scaling layer."""
    def __init__(self, scale: torch.Tensor, shift: torch.Tensor):
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class ResidualFraudLayer(nn.Module):
    """Fully‑connected layer with a residual skip connection."""
    def __init__(self, params: FraudLayerParameters, clip: bool = True):
        super().__init__()
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]],
                              dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scaler = FraudFeatureScaler(
            scale=torch.tensor(params.displacement_r, dtype=torch.float32),
            shift=torch.tensor(params.displacement_phi, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = self.scaler(out)
        return x + out  # residual


class FraudDetectionEnhanced(nn.Module):
    """Full fraud detection pipeline with residual layers and a final classifier."""
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]):
        super().__init__()
        modules: list[nn.Module] = [ResidualFraudLayer(input_params, clip=False)]
        modules.extend(ResidualFraudLayer(p, clip=True) for p in layers)
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @classmethod
    def build_fraud_detection_program(cls, input_params: FraudLayerParameters,
                                      layers: Iterable[FraudLayerParameters]) -> "FraudDetectionEnhanced":
        """Convenience constructor mirroring the original API."""
        return cls(input_params, layers)
