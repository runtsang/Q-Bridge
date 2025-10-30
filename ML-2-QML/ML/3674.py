"""Hybrid estimator with classical photonic preprocessing and a QNN head.

The module defines:

* :class:`FraudLayerParameters` –  hyper‑parameters for each layer.
* :func:`build_fraud_detection_program_clf` – builds a PyTorch
  sequential model that mirrors the photonic circuit.
* :class:`HybridEstimatorNN` – a regression network that first
  applies the fraud‑detection layers and then a simple fully‑connected
  head.  It can be used directly with PyTorch training loops.

The design keeps the classical core lightweight (only torch) while
offering an interface that can be swapped with a quantum head without
changing the training logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic or classical layer."""
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


def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            y = self.activation(self.linear(x))
            return y * self.scale + self.shift

    return Layer()


def build_fraud_detection_program_clf(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a PyTorch sequential model that mirrors the photonic
    fraud‑detection network.
    """
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class EstimatorNN(nn.Module):
    """Base QNN‐style regressor used as the head of the hybrid model."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class HybridEstimatorNN(nn.Module):
    """Full hybrid estimator: classical preprocessing + QNN head."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.pre = build_fraud_detection_program_clf(input_params, layers)
        self.head = EstimatorNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.head(self.pre(x))


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program_clf",
    "EstimatorNN",
    "HybridEstimatorNN",
]
