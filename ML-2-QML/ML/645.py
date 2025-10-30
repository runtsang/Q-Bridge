"""
Classical fraud‑detection model with enhanced regularisation and classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

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


class FraudDetectionModel(nn.Module):
    """
    A PyTorch implementation of the fraud‑detection pipeline.
    The network mirrors the layered structure of the photonic circuit
    but adds BatchNorm, Dropout and a final linear classifier.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [self._layer_from_params(input_params, clip=False)]
            + [self._layer_from_params(p, clip=True) for p in layers]
        )
        self.classifier = nn.Linear(2, num_classes)

    def _layer_from_params(self, params: FraudLayerParameters, clip: bool) -> nn.Module:
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi], [params.squeeze_r[0], params.squeeze_r[1]]],
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

        bn = nn.BatchNorm1d(2)
        dropout = nn.Dropout(p=0.1)
        activation = nn.Tanh()
        scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

        class Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = linear
                self.bn = bn
                self.activation = activation
                self.dropout = dropout
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
                y = self.linear(x)
                y = self.bn(y)
                y = self.activation(y)
                y = self.dropout(y)
                y = y * self.scale + self.shift
                return y

        return Layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
