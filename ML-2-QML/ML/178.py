"""Enhanced fraud detection model with residual connections and parameter conversion.

The class `FraudDetectionModel` mirrors the photonic architecture of the seed
but augments it with a lightweight residual block and a helper to export
parameters for the quantum counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic layer adapted to a classical network."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


class FraudLayer(nn.Module):
    """A single layer that emulates the photonic operations in a classical way."""

    def __init__(self, params: FraudLayerParameters, clip: bool = False) -> None:
        super().__init__()
        self.params = params
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
        self.scale = nn.Parameter(
            torch.tensor(params.displacement_r, dtype=torch.float32), requires_grad=False
        )
        self.shift = nn.Parameter(
            torch.tensor(params.displacement_phi, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        return out * self.scale + self.shift


class ResidualBlock(nn.Module):
    """Residual connection that adds the input to the output of a FraudLayer."""

    def __init__(self, params: FraudLayerParameters) -> None:
        super().__init__()
        self.params = params
        self.layer = FraudLayer(params, clip=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer(x)


class FraudDetectionModel(nn.Module):
    """Full fraud‑detection network with an initial layer and a stack of residual blocks."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        residual_params: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.input_layer = FraudLayer(input_params, clip=False)
        self.residuals = nn.ModuleList(
            [ResidualBlock(p) for p in residual_params]
        )
        self.output = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for res in self.residuals:
            out = res(out)
        return self.output(out)

    def to_quantum_params(self) -> List[FraudLayerParameters]:
        """
        Convert the parameters of the classical network into a list
        that can be fed into the quantum circuit.
        """
        return [self.input_layer.params] + [res.params for res in self.residuals]
