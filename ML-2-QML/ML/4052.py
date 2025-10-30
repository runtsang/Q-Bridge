"""Hybrid fraud detection model combining CNN, photonic-inspired layers and regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

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
    bias: float = 0.0


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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


class CNNFeatureExtractor(nn.Module):
    """Simple 2‑D CNN that mirrors Quantum‑NAT’s feature extractor."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        # Global average pooling to 16 channels
        x = torch.mean(x, dim=[2, 3])
        return self.fc(x)


class EstimatorRegressor(nn.Module):
    """Regression head identical to EstimatorQNN but with a single output."""

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


class FraudDetectionHybridModel(nn.Module):
    """Full hybrid model: CNN → photonic‑inspired layers → regression."""

    def __init__(
        self,
        cnn: nn.Module,
        quantum_params: FraudLayerParameters,
        quantum_layers: Iterable[FraudLayerParameters],
        regressor: nn.Module,
    ) -> None:
        super().__init__()
        self.cnn = cnn
        self.quantum = nn.Sequential(
            _layer_from_params(quantum_params, clip=False),
            *(_layer_from_params(lp, clip=True) for lp in quantum_layers),
        )
        self.regressor = regressor

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.cnn(x)
        quantum_out = self.quantum(features)
        return self.regressor(quantum_out)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> FraudDetectionHybridModel:
    """Convenience factory that stitches together the CNN, quantum layers and regressor."""
    return FraudDetectionHybridModel(
        cnn=CNNFeatureExtractor(),
        quantum_params=input_params,
        quantum_layers=layers,
        regressor=EstimatorRegressor(),
    )


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybridModel",
]
