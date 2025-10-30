from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable

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

class FraudDetectionHybrid(nn.Module):
    """
    Classical fraud detection model that mirrors a photonic circuit and
    appends a small feed‑forward head, enabling comparison with the
    quantum implementation.
    """

    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.feature_extractor = self._build_feature_extractor(input_params, layers)
        self.head = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def _build_feature_extractor(self,
                                 input_params: FraudLayerParameters,
                                 layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
        modules = [self._layer_from_params(input_params, clip=False)]
        modules += [self._layer_from_params(p, clip=True) for p in layers]
        return nn.Sequential(*modules)

    def _layer_from_params(self,
                           params: FraudLayerParameters,
                           *,
                           clip: bool) -> nn.Module:
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
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.activation(self.linear(x))
                out = out * self.scale + self.shift
                return out

        return Layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.head(features)

__all__ = ["FraudDetectionHybrid"]
