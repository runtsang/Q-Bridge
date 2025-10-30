"""Hybrid classical layer that fuses a fully‑connected quantum‑style interface with a photonic fraud‑detection backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn


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


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _linear_layer(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_linear_layer(input_params, clip=False)]
    modules.extend(_linear_layer(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class HybridLayer(nn.Module):
    """Hybrid fully‑connected layer that mimics a quantum circuit while retaining a classical neural‑network backbone."""

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.model = build_fraud_detection_program(input_params, layers)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Accept a list of parameters and return the model output as a NumPy array."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        output = self.model(values)
        return output.detach().numpy()


def FCL() -> HybridLayer:
    """Return an instance of the hybrid layer."""
    input_params = FraudLayerParameters(
        bs_theta=0.1,
        bs_phi=0.2,
        phases=(0.3, 0.4),
        squeeze_r=(0.5, 0.6),
        squeeze_phi=(0.7, 0.8),
        displacement_r=(0.9, 1.0),
        displacement_phi=(1.1, 1.2),
        kerr=(1.3, 1.4),
    )
    layers = [
        FraudLayerParameters(
            bs_theta=0.2,
            bs_phi=0.3,
            phases=(0.4, 0.5),
            squeeze_r=(0.6, 0.7),
            squeeze_phi=(0.8, 0.9),
            displacement_r=(1.0, 1.1),
            displacement_phi=(1.2, 1.3),
            kerr=(1.4, 1.5),
        )
    ]
    return HybridLayer(input_params, layers)


__all__ = ["HybridLayer", "FCL", "FraudLayerParameters"]
