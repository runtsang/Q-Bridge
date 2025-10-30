"""Classical hybrid model combining a quanvolution filter and fraud-detection inspired head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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

def _build_fraud_layer(params: FraudLayerParameters, clip: bool = False) -> nn.Module:
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

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_fraud_detection_program(input_params: FraudLayerParameters,
                                 layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
    modules = [_build_fraud_layer(input_params, clip=False)]
    modules.extend(_build_fraud_layer(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class QuanvolutionHybridModel(nn.Module):
    """Classical hybrid model: a 2×2 convolution filter followed by a fraud‑detection style head."""
    def __init__(self,
                 fraud_layers: Iterable[FraudLayerParameters] | None = None):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        feature_dim = 4 * 14 * 14
        self.fc = nn.Linear(feature_dim, 10)
        if fraud_layers is None:
            self.fraud_head = nn.Linear(10, 1)
        else:
            dummy = FraudLayerParameters(0.0, 0.0, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
                                         (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
            self.fraud_head = build_fraud_detection_program(dummy, fraud_layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        flat = features.view(x.size(0), -1)
        logits = self.fc(flat)
        fraud_out = self.fraud_head(logits)
        return F.log_softmax(fraud_out, dim=-1)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "QuanvolutionHybridModel"]
