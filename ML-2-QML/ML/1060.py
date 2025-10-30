"""Enhanced classical fraud detection model with residual connections and dropout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

class FraudDetectionModel(nn.Module):
    """Classical neural network mirroring the photonic fraud detection circuit.

    Adds residual connections, optional batch‑norm, and dropout for better
    generalisation while preserving the layered structure of the seed.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        modules = [self._layer_from_params(input_params, clip=False, residual=False)]
        modules.extend(
            self._layer_from_params(layer, clip=True, residual=True)
            for layer in layers
        )
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    def _layer_from_params(
        self,
        params: FraudLayerParameters,
        clip: bool,
        residual: bool,
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
        scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

        class Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = linear
                self.activation = activation
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)
                if use_batchnorm:
                    self.bn = nn.BatchNorm1d(2)
                if dropout > 0.0:
                    self.drop = nn.Dropout(dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.activation(self.linear(x))
                out = out * self.scale + self.shift
                if use_batchnorm:
                    out = self.bn(out)
                if dropout > 0.0:
                    out = self.drop(out)
                if residual:
                    out = out + x
                return out

        return Layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
