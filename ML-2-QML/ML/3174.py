from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Iterable, Tuple

# Dataset generation
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# Fraud‑style layer parameters
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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

# Dataset that can optionally apply fraud‑style transformation
class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int, fraud_layers: Iterable[FraudLayerParameters] | None = None):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.fraud_transform = None
        if fraud_layers is not None:
            self.fraud_transform = build_fraud_detection_program(
                FraudLayerParameters(*self._default_params()), fraud_layers
            )

    def _default_params(self) -> Tuple[float, float, Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        return (0.0, 0.0, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        state = torch.tensor(self.features[index], dtype=torch.float32)
        if self.fraud_transform is not None:
            state = self.fraud_transform(state)
        return {
            "states": state.squeeze(-1) if state.ndim > 1 else state,
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# Classical regression model that optionally prepends fraud‑style layers
class RegressionModel(nn.Module):
    def __init__(self, num_features: int, fraud_layers: Iterable[FraudLayerParameters] | None = None):
        super().__init__()
        if fraud_layers is not None:
            self.fraud_transform = build_fraud_detection_program(
                FraudLayerParameters(*self._default_params()), fraud_layers
            )
            input_dim = 2
        else:
            self.fraud_transform = None
            input_dim = num_features

        self.base = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def _default_params(self) -> Tuple[float, float, Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        return (0.0, 0.0, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        if self.fraud_transform is not None:
            state_batch = self.fraud_transform(state_batch)
        return self.base(state_batch).squeeze(-1)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "RegressionDataset",
    "RegressionModel",
    "generate_superposition_data",
]
