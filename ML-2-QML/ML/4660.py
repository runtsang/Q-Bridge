from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import numpy as np

# --------------------------------------------------------------------------- #
# Classical fraud‑detection primitives
# --------------------------------------------------------------------------- #

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


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# Hybrid module combining classical layers with classical soft‑max
# --------------------------------------------------------------------------- #

class FraudDetectionHybrid(nn.Module):
    """
    A hybrid fraud‑detection model that ends with a soft‑max layer
    (mimicking the SamplerQNN output) and optionally a final
    expectation layer (mimicking the FCL output).
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        use_softmax: bool = True,
        use_expectation: bool = False
    ) -> None:
        super().__init__()
        self.base = build_fraud_detection_program(input_params, layers)
        # Optional soft‑max output
        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(dim=-1) if use_softmax else nn.Identity()
        # Optional expectation layer
        self.use_expectation = use_expectation
        if use_expectation:
            # a single linear + tanh that mimics FCL expectation
            self.expect = nn.Sequential(
                nn.Linear(1, 1),
                nn.Tanh()
            )
        else:
            self.expect = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.base(inputs)
        x = self.softmax(x)
        x = self.expect(x)
        return x


__all__ = [
    "FraudLayerParameters",
    "_layer_from_params",
    "build_fraud_detection_program",
    "FraudDetectionHybrid"
]
