# FraudDetectionHybrid.py
"""Hybrid classical‑quantum fraud detection model.

The module defines:

* ``FraudLayerParameters`` – the same parameter container used in the photonic example.
* ``build_fraud_detection_program`` – constructs a PyTorch sequential model that mirrors the
  photonic circuit, with optional clipping of the parameters.
* ``EstimatorNN`` – a tiny feed‑forward regressor inspired by the EstimatorQNN example.
* ``FraudDetectionHybrid`` – a single ``nn.Module`` that first processes the input
  through the classical layers and then feeds the result into the estimator network.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Tuple

# --------------------------------------------------------------------------- #
# 1.  Classical layer definitions (photonic‑inspired)
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to ``[-bound, bound]``."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Return a single two‑node ``nn.Module`` that replicates the photonic layer."""
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
# 2.  Tiny estimator network (classical feed‑forward)
# --------------------------------------------------------------------------- #
class EstimatorNN(nn.Module):
    """Tiny regression network used as the final quantum‑inspired estimator."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)


# --------------------------------------------------------------------------- #
# 3.  Hybrid model that stitches classical layers with estimator
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid(nn.Module):
    """Hybrid model that first processes the input with photonic‑inspired
    classical layers and then applies a classical estimator network.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.classical = build_fraud_detection_program(input_params, layers)
        self.estimator = EstimatorNN()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass through the classical layers followed by the estimator."""
        x = self.classical(inputs)
        return self.estimator(x)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "EstimatorNN",
    "FraudDetectionHybrid",
]
