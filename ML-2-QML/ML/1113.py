"""Enhanced classical fraud‑detection model with residuals and regularisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


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
    """Clamp a scalar to the interval [−bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Build a single linear block with optional clipping."""
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)

    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    block = nn.Sequential(
        linear,
        nn.BatchNorm1d(2),
        nn.Tanh(),
        nn.Dropout(p=0.2),
        nn.Linear(2, 2)  # second linear to allow residual scaling
    )
    return block


class FraudDetector(nn.Module):
    """
    A PyTorch model that mirrors the layered photonic structure.
    Residual connections are added between successive layers.
    """
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.input_block = _layer_from_params(input_params, clip=False)
        self.blocks = nn.ModuleList()
        for layer in layers:
            self.blocks.append(_layer_from_params(layer, clip=True))
        self.output_layer = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_block(x)
        for block in self.blocks:
            residual = out
            out = block(out)
            out = out + residual  # residual connection
        out = self.output_layer(out)
        return torch.sigmoid(out)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> FraudDetector:
    """Convenience wrapper to instantiate the FraudDetector."""
    return FraudDetector(input_params, layers)


__all__ = ["FraudLayerParameters", "FraudDetector"]
