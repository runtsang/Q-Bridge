"""Hybrid classical estimator mirroring the photonic fraud‑detection architecture.

The network consists of a sequence of parameterised linear layers with custom
activation, scaling, and clipping, followed by a final linear regression head.
It is deliberately structured to be a classical analogue of the quantum circuit
in the QML module, enabling side‑by‑side experiments and comparative training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a single fully‑connected layer."""
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


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single layer with custom weight, bias, activation, scaling and shift."""
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Build the full classical surrogate network."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class EstimatorQNN(nn.Module):
    """Hybrid estimator that mimics the quantum circuit with a classical surrogate.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first layer (no clipping).
    hidden_layers : Iterable[FraudLayerParameters]
        Parameters for subsequent layers (clipped).
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.model = build_fraud_detection_program(input_params, hidden_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "EstimatorQNN"]
