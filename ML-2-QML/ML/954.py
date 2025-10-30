"""Hybrid fraud‑detection model with classical residual blocks and optional
dropout / batch‑norm.  The interface mirrors the original seed but adds
extra hyper‑parameters for richer experimentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    # New hyper‑parameters
    dropout_rate: float = 0.0
    batchnorm: bool = False


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class _ResidualBlock(nn.Module):
    """A residual block that optionally applies batch‑norm and dropout."""
    def __init__(self, params: FraudLayerParameters, *, clip: bool) -> None:
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32,
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

        self.dropout = nn.Dropout(params.dropout_rate) if params.dropout_rate > 0 else nn.Identity()
        self.bn = nn.BatchNorm1d(2) if params.batchnorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        out = self.dropout(out)
        out = self.bn(out)
        return out + x  # residual addition


def build_fraud_detection_model(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Construct a residual network mirroring the layered structure."""
    modules: list[nn.Module] = [_ResidualBlock(input_params, clip=False)]
    modules += [_ResidualBlock(layer, clip=True) for layer in layers]
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


__all__ = ["FraudLayerParameters", "build_fraud_detection_model"]
