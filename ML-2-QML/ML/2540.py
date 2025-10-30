from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(x))
            out = out * self.scale + self.shift
            return out

    return Layer()


def build_fraud_detection_model(
    input_params: FraudLayerParameters,
    layers: list[FraudLayerParameters],
    final_features: int = 1,
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, final_features))
    return nn.Sequential(*modules)


def generate_fraud_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = (np.sin(angles) + 0.1 * np.cos(2 * angles)) > 0.0
    return X, y.astype(np.float32)


class FraudDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_fraud_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class FraudDetectionModel(nn.Module):
    def __init__(self, num_features: int, num_layers: int = 3):
        super().__init__()
        self.params = [
            FraudLayerParameters(
                bs_theta=np.random.rand(),
                bs_phi=np.random.rand(),
                phases=(np.random.rand(), np.random.rand()),
                squeeze_r=(np.random.rand(), np.random.rand()),
                squeeze_phi=(np.random.rand(), np.random.rand()),
                displacement_r=(np.random.rand(), np.random.rand()),
                displacement_phi=(np.random.rand(), np.random.rand()),
                kerr=(np.random.rand(), np.random.rand()),
            )
            for _ in range(num_layers + 1)
        ]  # input + hidden layers
        self.model = build_fraud_detection_model(self.params[0], self.params[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)
