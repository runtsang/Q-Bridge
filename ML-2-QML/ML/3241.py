from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FraudLayerParameters:
    """
    Parameters describing a fully connected layer in the classical model.
    """
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
    """
    Create a sequential PyTorch model mirroring the layered structure.
    """
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudDetectionHybrid(nn.Module):
    """
    Classical component of the hybrid fraud detection model.
    Builds a feed‑forward network from photonic‑style parameters
    and produces a soft‑max probability vector.
    """

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]):
        super().__init__()
        self.network = build_fraud_detection_program(input_params, layers)
        self.prob_layer = nn.Sequential(
            nn.Linear(1, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return fraud probability distribution."""
        features = self.network(x)
        return self.prob_layer(features)

    def get_quantum_params(self) -> tuple[FraudLayerParameters, Sequence[FraudLayerParameters]]:
        """
        Return the parameters that will be fed into the quantum sampler.
        The first set corresponds to input parameters (2) and the second
        set to weight parameters (4) derived from the network weights.
        """
        # For illustration, we simply map the first two rows of the first
        # layer's weight matrix to the input parameters and the next
        # four rows to the weight parameters of the sampler.
        first_layer: nn.Linear = self.network[0].linear
        weight = first_layer.weight.detach()
        bias = first_layer.bias.detach()
        # Input params
        input_params = FraudLayerParameters(
            bs_theta=float(weight[0, 0]),
            bs_phi=float(weight[0, 1]),
            phases=(float(bias[0]), float(bias[1])),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.0, 0.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        # Weight params
        weight_params = [
            FraudLayerParameters(
                bs_theta=float(weight[1, 0]),
                bs_phi=float(weight[1, 1]),
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
        ]
        return input_params, weight_params


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid"]
