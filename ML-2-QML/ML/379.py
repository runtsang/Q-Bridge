"""Classical fraud detection model with enhanced regularisation.

This module extends the original seed by adding dropout and batch‑norm
after each photonic‑inspired linear layer.  The resulting model
mirrors the Strawberry Fields circuit but offers additional
regularisation and a clean PyTorch interface.
"""

from __future__ import annotations

import torch
from torch import nn
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


class FraudDetectionModel(nn.Module):
    """Hybrid fraud detection model built from photonic‑inspired layers."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout: float = 0.1,
        clip: bool = True,
    ) -> None:
        super().__init__()
        modules = [
            self._layer_from_params(input_params, clip=False, dropout=dropout)
        ]
        modules += [
            self._layer_from_params(p, clip=clip, dropout=dropout)
            for p in layers
        ]
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    @staticmethod
    def _clip(value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _layer_from_params(
        self,
        params: FraudLayerParameters,
        *,
        clip: bool,
        dropout: float,
    ) -> nn.Module:
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
                self.dropout = nn.Dropout(dropout)
                self.bn = nn.BatchNorm1d(2)
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                out = self.linear(inputs)
                out = self.activation(out)
                out = self.bn(out)
                out = self.dropout(out)
                out = out * self.scale + self.shift
                return out

        return Layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
