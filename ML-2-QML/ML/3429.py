"""Hybrid fraud‑detection module: classical part.

This module keeps the classical photonic‑style network but now wrapped in a
class that can be instantiated independently or used through the static
methods.  The interface is deliberately simple: the caller passes a list of
FraudLayerParameters and receives a PyTorch nn.Sequential model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn


# --------------------------------------------------------------------------- #
# 1. Classical parameter container – identical to the original seed
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


# --------------------------------------------------------------------------- #
# 2. Helper: build a single linear block with optional clipping
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Build a tiny linear‑tanh‑scale block that mimics the photonic layer."""
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


# --------------------------------------------------------------------------- #
# 3. Build the sequential classical stack (first layer un‑clipped)
# --------------------------------------------------------------------------- #
def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Return a PyTorch Sequential model mirroring the photonic architecture."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# 4. Public API – same __all__ as the original
# --------------------------------------------------------------------------- #
class FraudGraphHybrid:
    """Facade for the classical fraud‑detection network.

    The class is intentionally lightweight; all heavy‑weight quantum logic
    lives in the companion QML module.
    """
    @staticmethod
    def build_classical_model(input_params: FraudLayerParameters,
                              layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
        """Create the classical feed‑forward network."""
        return build_fraud_detection_program(input_params, layers)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudGraphHybrid"]
