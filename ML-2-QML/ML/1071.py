"""Classical neural network mirroring the photonic fraud detection architecture with dropout regularization."""
from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Tuple

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

class FraudDetectionHybrid:
    """Classical neural network mirroring the photonic fraud detection architecture."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 dropout: float = 0.0,
                 clip: bool = True) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.dropout = dropout
        self.clip = clip
        self.model = self._build_model()

    def _layer_from_params(self,
                           params: FraudLayerParameters,
                           *,
                           clip: bool,
                           dropout: float) -> nn.Module:
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]],
                              dtype=torch.float32)
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
                self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.activation(self.linear(x))
                y = y * self.scale + self.shift
                return self.dropout(y)

        return Layer()

    def _build_model(self) -> nn.Sequential:
        modules = [self._layer_from_params(self.input_params, clip=False, dropout=self.dropout)]
        modules += [self._layer_from_params(layer, clip=self.clip, dropout=self.dropout)
                    for layer in self.layers]
        modules.append(nn.Linear(2, 1))
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
