"""Hybrid classical model combining CNN, fraud‑detection layers, and QCNN‑style MLP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Fraud‑Detection style parameters and helpers
# ----------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the classical model."""
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


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single layer mirroring a photonic block."""
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
    """Construct the fraud‑detection MLP as a sequential module."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# ----------------------------------------------------------------------
# QCNN‑style fully‑connected block
# ----------------------------------------------------------------------
class QCNNHybrid(nn.Module):
    """QCNN‑inspired stack operating on a 4‑dimensional vector."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(4, 8), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(8, 5), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(5, 4), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(4, 3), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(3, 3), nn.Tanh())
        self.head = nn.Linear(3, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


# ----------------------------------------------------------------------
# Main hybrid model
# ----------------------------------------------------------------------
class QuantumHybridModel(nn.Module):
    """Full classical hybrid model that stitches together CNN, fraud layer, and QCNN."""

    def __init__(
        self,
        fraud_input_params: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully‑connected projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)
        # Fraud‑detection style layers
        self.fraud = build_fraud_detection_program(fraud_input_params, fraud_layers)
        # QCNN‑style MLP
        self.qcnn = QCNNHybrid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        # CNN + projection
        features = self.features(x)
        flattened = features.view(bsz, -1)
        projected = self.fc(flattened)
        projected = self.norm(projected)
        # Fraud layers
        fraud_out = self.fraud(projected)
        # QCNN stack producing a scalar output
        out = self.qcnn(fraud_out)
        return out.reshape(-1, 1)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "QCNNHybrid", "QuantumHybridModel"]
