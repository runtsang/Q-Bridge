"""Classical regression model with fraud‑detection style feature extractor and probabilistic sampler."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Iterable, Tuple

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
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
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

def SamplerQNN() -> nn.Module:
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return nn.functional.softmax(self.net(inputs), dim=-1)

    return SamplerModule()

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class SharedRegressionModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        if fraud_params is not None:
            self.fraud_head = build_fraud_detection_program(
                FraudLayerParameters(
                    bs_theta=0.0, bs_phi=0.0, phases=(0.0, 0.0),
                    squeeze_r=(0.0, 0.0), squeeze_phi=(0.0, 0.0),
                    displacement_r=(0.0, 0.0), displacement_phi=(0.0, 0.0),
                    kerr=(0.0, 0.0),
                ),
                fraud_params,
            )
        else:
            self.fraud_head = nn.Identity()
        self.sampler = SamplerQNN()
        self.regression_head = nn.Linear(2, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(state_batch)
        x = self.fraud_head(x)
        probs = self.sampler(x)
        y = probs[:, 0]
        return self.regression_head(y.unsqueeze(-1)).squeeze(-1)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "SamplerQNN",
    "generate_superposition_data",
    "RegressionDataset",
    "SharedRegressionModel",
]
