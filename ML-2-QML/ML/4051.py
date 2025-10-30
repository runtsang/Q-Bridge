# Hybrid QCNN model combining convolutional, regression, and fraud-detection inspired layers.
# This module defines HybridQCNNModel, a PyTorch model that mirrors the structure of the original QCNN
# but replaces the fully‑connected blocks with a stack of parameterised “Fraud” layers that emulate photonic mixing and squeezing.
# The same helper functions used for generating superposition data are reused to create a regression dataset that can be fed into either the classical or quantum branch.

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    features = np.concatenate([np.cos(thetas)[:, None], np.sin(thetas)[:, None] * np.exp(1j * phis)], axis=1)
    features = np.real(features).astype(np.float32)
    labels = (np.sin(2 * thetas) * np.cos(phis)).astype(np.float32)
    return features, labels

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

@dataclass
class FraudLayerParameters:
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

def build_fraud_detection_model(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class HybridQCNNModel(nn.Module):
    def __init__(self, num_features: int = 8, num_layers: int = 3):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(num_features, 16), nn.Tanh())

        self.conv_layers: list[nn.Sequential] = []
        for _ in range(num_layers):
            params = FraudLayerParameters(
                bs_theta=np.random.uniform(-np.pi, np.pi),
                bs_phi=np.random.uniform(-np.pi, np.pi),
                phases=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
                squeeze_r=(np.random.uniform(0, 1), np.random.uniform(0, 1)),
                squeeze_phi=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
                displacement_r=(np.random.uniform(0, 1), np.random.uniform(0, 1)),
                displacement_phi=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
                kerr=(np.random.uniform(-1, 1), np.random.uniform(-1, 1)),
            )
            self.conv_layers.append(build_fraud_detection_model(params, []))

        self.pool = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.head = nn.Linear(12, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        for conv in self.conv_layers:
            x = conv(x)
        x = self.pool(x)
        return torch.sigmoid(self.head(x))

def QCNN() -> HybridQCNNModel:
    return HybridQCNNModel(num_features=8, num_layers=3)

__all__ = ["HybridQCNNModel", "RegressionDataset", "QCNN", "FraudLayerParameters", "build_fraud_detection_model"]
