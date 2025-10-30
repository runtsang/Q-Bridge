"""Hybrid model combining QCNN feature extractor and photonic fraud detection."""
from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence

# ----------------------------------------------------
# Photonic‑style layer definitions (classical analogue)
# ----------------------------------------------------
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
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ----------------------------------------------------
# QCNN‑style feature extractor (enhanced classical network)
# ----------------------------------------------------
class QCNNFeatureExtractor(nn.Module):
    """Enhanced QCNN inspired network with batch‑norm, dropout and ReLU."""
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, 32), nn.BatchNorm1d(32), nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.1)
        )
        self.pool1 = nn.Sequential(
            nn.Linear(32, 24), nn.BatchNorm1d(24), nn.ReLU(), nn.Dropout(0.1)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(24, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.1)
        )
        self.pool2 = nn.Sequential(
            nn.Linear(16, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(0.1)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(8, 4), nn.BatchNorm1d(4), nn.ReLU()
        )
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        return torch.sigmoid(self.head(x))

# ----------------------------------------------------
# Hybrid model (classical side)
# ----------------------------------------------------
class HybridQCNNFraudModel(nn.Module):
    """Hybrid classical model: QCNN feature extractor + photonic fraud detection."""
    def __init__(
        self,
        input_dim: int,
        fraud_input_params: FraudLayerParameters,
        fraud_hidden_params: Iterable[FraudLayerParameters],
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.qcnn = QCNNFeatureExtractor(input_dim)
        self.fraud = build_fraud_detection_program(fraud_input_params, fraud_hidden_params)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.qcnn(inputs)
        return self.fraud(features)

def QCNNFraudFactory(
    input_dim: int,
    fraud_input_params: FraudLayerParameters,
    fraud_hidden_params: Iterable[FraudLayerParameters],
) -> HybridQCNNFraudModel:
    """Convenience factory returning a ready‑to‑train model."""
    return HybridQCNNFraudModel(input_dim, fraud_input_params, fraud_hidden_params)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "QCNNFeatureExtractor",
    "HybridQCNNFraudModel",
    "QCNNFraudFactory",
]
