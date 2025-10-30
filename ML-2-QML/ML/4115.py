"""Hybrid classical model that emulates quanvolution with fraud‑style scaling."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional

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

def build_fraud_detection_head(
    params: List[FraudLayerParameters],
    final_out: int = 1,
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(params[0], clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in params[1:])
    modules.append(nn.Linear(2, final_out))
    return nn.Sequential(*modules)

class QuanvolutionHybrid(nn.Module):
    """Classical hybrid model: 2×2 image patches → convolution → fraud‑style linear layers."""
    def __init__(
        self,
        conv_out_channels: int = 4,
        fraud_params: Optional[List[FraudLayerParameters]] = None,
        final_out: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, conv_out_channels, kernel_size=2, stride=2)
        if fraud_params:
            self.head = build_fraud_detection_head(fraud_params, final_out)
        else:
            # Default simple regressor
            self.head = nn.Sequential(
                nn.Linear(conv_out_channels * 14 * 14, 8),
                nn.Tanh(),
                nn.Linear(8, final_out),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        features = self.conv(x)
        flat = features.view(features.size(0), -1)
        logits = self.head(flat)
        return logits

__all__ = ["QuanvolutionHybrid", "FraudLayerParameters", "build_fraud_detection_head"]
