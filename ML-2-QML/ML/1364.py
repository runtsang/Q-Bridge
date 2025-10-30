"""Enhanced classical fraud‑detection model with attention and regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FraudLayerParameters:
    """Parameters for a single fully‑connected block."""
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


def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
    dropout_rate: float,
) -> nn.Module:
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
    scale = nn.Parameter(torch.tensor(params.displacement_r, dtype=torch.float32))
    shift = nn.Parameter(torch.tensor(params.displacement_phi, dtype=torch.float32))

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.scale = scale
            self.shift = shift
            self.attention = nn.Linear(2, 2)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.linear(inputs)
            out = self.activation(out)
            att = F.softmax(self.attention(out), dim=1)
            out = att * out
            out = out * self.scale + self.shift
            out = self.dropout(out)
            return out

    return Layer()


class FraudDetectionEnhanced(nn.Module):
    """Classical fraud‑detection model with optional attention and dropout."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dropout_rate: float = 0.1,
        reg_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.reg_weight = reg_weight
        self.layers = nn.ModuleList()
        self.layers.append(_layer_from_params(input_params, clip=False, dropout_rate=dropout_rate))
        for layer_params in layers:
            self.layers.append(_layer_from_params(layer_params, clip=True, dropout_rate=dropout_rate))
        self.output = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

    def regularization(self) -> torch.Tensor:
        """Return L2 regularization term over all linear weights."""
        reg = 0.0
        for layer in self.layers:
            reg += torch.norm(layer.linear.weight, p=2)
        return self.reg_weight * reg

    def early_stop(self, loss_history: Sequence[float], patience: int = 5) -> bool:
        """Simple early‑stop: stop if loss has not improved for `patience` epochs."""
        if len(loss_history) < patience + 1:
            return False
        return all(loss_history[-i] >= loss_history[-i - 1] for i in range(1, patience + 1))


__all__ = ["FraudLayerParameters", "FraudDetectionEnhanced"]
