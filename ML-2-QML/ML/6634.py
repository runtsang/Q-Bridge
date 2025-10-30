"""Enhanced classical fraud‑detection model with dropout, batch norm, and quantum‑aware loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


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
    """Clamp values to a safe range before they are used in the layer."""
    return max(-bound, min(bound, value))


class ScaleShift(nn.Module):
    """Element‑wise scale and shift used as the final layer of each photonic
    analogue block."""

    def __init__(self, scale: torch.Tensor, shift: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
    dropout_rate: float = 0.0,
    batch_norm: bool = False,
) -> nn.Module:
    """Create a fully‑connected layer that optionally includes dropout and
    batch‑normalisation."""
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
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

    layers: list[nn.Module] = [linear, activation]
    if batch_norm:
        layers.append(nn.BatchNorm1d(2))
    if dropout_rate > 0.0:
        layers.append(nn.Dropout(dropout_rate))
    layers.append(ScaleShift(scale, shift))

    return nn.Sequential(*layers)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *,
    dropout_rate: float = 0.0,
    batch_norm: bool = False,
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [
        _layer_from_params(
            input_params, clip=False, dropout_rate=dropout_rate, batch_norm=batch_norm
        )
    ]
    modules.extend(
        _layer_from_params(
            layer, clip=True, dropout_rate=dropout_rate, batch_norm=batch_norm
        )
        for layer in layers
    )
    modules.append(nn.Linear(2, 1))
    modules.append(nn.Sigmoid())
    return nn.Sequential(*modules)


class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud detection model that can incorporate a quantum sub‑module."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        quantum_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.classical = build_fraud_detection_program(
            input_params, layers, dropout_rate=dropout_rate, batch_norm=batch_norm
        )
        self.quantum_module = quantum_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.classical(x)
        if self.quantum_module is not None:
            qy = self.quantum_module(x)
            y = (y + qy) / 2.0
        return y

    def loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        quantum_penalty: float = 0.0,
    ) -> torch.Tensor:
        """Binary cross‑entropy loss with an optional quantum‑aware penalty."""
        bce = F.binary_cross_entropy(predictions, targets)
        if self.quantum_module is not None and quantum_penalty > 0.0:
            # Use a dummy input to probe the quantum module; the penalty is
            # the deviation of its output from 0.5 (neutral risk).
            q_out = self.quantum_module(torch.tensor([0.0, 0.0]))
            q_loss = (q_out - 0.5).abs()
            return bce + quantum_penalty * q_loss
        return bce


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybrid",
]
