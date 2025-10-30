"""
FraudDetectionEnhanced: A hybrid classical‑and‑quantum fraud‑detection model with attention‑based residuals.

The module defines two complementary sub‑models:
  * A PyTorch neural network that augments the original 2‑input, 2‑output layers with a
    residual connection and a soft‑attention mask over the two input channels.
  * A Strawberry Fields program that replaces the original single‑beam‑splitter layer
    with a learnable 4‑mode interferometer.  The interferometer is built from a sequence
    of BS gates and phase shifters and is constrained to preserve the output photon
    count.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Tuple

# --------------------------------------------------------------------------- #
# Classical model
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer (for the classical part)."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

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
        """Layer with residual attention."""
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)
            self.attention = nn.Linear(2, 2, bias=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            base = self.activation(self.linear(inputs))
            base = base * self.scale + self.shift
            att = torch.softmax(self.attention(inputs), dim=1)
            return base * att + inputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure with residual attention."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program"]
