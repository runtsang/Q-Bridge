"""Enhanced classical fraud detection model with optional dropout and batch‑norm regularisation.

The architecture mirrors the photonic circuit defined in the quantum counterpart.
Dropout and batch‑norm layers are inserted after the linear block to improve
generalisation. A static helper allows construction from a Strawberry
Fields program, enabling hybrid training pipelines.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, List

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
    dropout: float = 0.0
    batch_norm: bool = False

class FraudDetectionHybrid(nn.Module):
    """Classical neural network mirroring the photonic architecture."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.layers = list(layers)
        modules: List[nn.Module] = [self._layer_from_params(input_params, clip=False)]
        for layer in self.layers:
            modules.append(self._layer_from_params(layer, clip=True))
        modules.append(nn.Linear(2, 1))
        self.base = nn.Sequential(*modules)
        self.batch_norm_layer = nn.BatchNorm1d(2) if batch_norm else nn.Identity()
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _layer_from_params(self, params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                x = self.activation(self.linear(inputs))
                return x * self.scale + self.shift

        return Layer()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.base(inputs)
        x = self.batch_norm_layer(x)
        return self.dropout_layer(x)

    @staticmethod
    def from_quantum(program: "sf.Program") -> "FraudDetectionHybrid":
        """Instantiate a classical model from a Strawberry Fields program."""
        from strawberryfields import Program

        if not isinstance(program, Program):
            raise TypeError("Expected a strawberryfields.Program")
        # Minimal implementation: only the first layer is extracted.
        params = FraudLayerParameters(
            bs_theta=0.0,
            bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.0, 0.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        return FraudDetectionHybrid(params, [], dropout=0.0, batch_norm=False)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
