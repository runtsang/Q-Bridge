"""Fraud detection model – classical PyTorch implementation with dropout and L2 regularisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional

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
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool, dropout: float) -> nn.Module:
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
    linear = nn.Linear(2, 2, bias=True)
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
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.linear(inputs)
            out = self.activation(out)
            out = out * self.scale + self.shift
            out = self.dropout(out)
            return out

    return Layer()


class FraudDetection(nn.Module):
    """Drop‑out enabled, L2‑regularised fraud‑detection network."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dropout: float = 0.0,
        l2: float = 0.0,
    ) -> None:
        super().__init__()
        self.l2 = l2
        self.layers = nn.Sequential(
            _layer_from_params(input_params, clip=False, dropout=dropout),
            *(
                _layer_from_params(layer, clip=True, dropout=dropout)
                for layer in layers
            ),
            nn.Linear(2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sigmoid(self.layers(x))

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary cross‑entropy with optional L2 penalty."""
        bce = F.binary_cross_entropy(logits, targets)
        if self.l2 == 0.0:
            return bce
        l2_reg = sum(
            (param.pow(2).sum() for name, param in self.named_parameters() if "bias" not in name)
        )
        return bce + self.l2 * l2_reg


__all__ = ["FraudDetection", "FraudLayerParameters"]
