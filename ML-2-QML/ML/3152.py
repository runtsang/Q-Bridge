from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
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

class FraudDetectionHybridModel(nn.Module):
    """
    Hybrid classical fraud detection model that combines photonic-inspired
    fully‑connected layers with QCNN‑style convolution and pooling.
    """
    def __init__(
        self,
        layers: Iterable[FraudLayerParameters],
        pool_every: int = 2,
        input_dim: int = 8,
    ) -> None:
        super().__init__()
        # Feature extractor resembling QCNN's 8‑to‑16 mapping
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        # Photonic layers
        self.layers = nn.ModuleList(
            [_layer_from_params(l, clip=True) for l in layers]
        )
        self.pool_every = pool_every
        # Pooling mimicking QCNN's dimensionality reduction
        self.pool = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.head = nn.Linear(12, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        for i, layer in enumerate(self.layers, 1):
            x = layer(x)
            if i % self.pool_every == 0:
                x = self.pool(x)
        return torch.sigmoid(self.head(x))

def FraudDetectionHybridModelFactory(
    num_layers: int,
    seed: int | None = None,
) -> FraudDetectionHybridModel:
    """
    Simple factory that generates a model with random photonic parameters.
    """
    import random
    if seed is not None:
        random.seed(seed)
    layers = [
        FraudLayerParameters(
            bs_theta=random.uniform(-np.pi, np.pi),
            bs_phi=random.uniform(-np.pi, np.pi),
            phases=(random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)),
            squeeze_r=(random.uniform(0, 1), random.uniform(0, 1)),
            squeeze_phi=(random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)),
            displacement_r=(random.uniform(0, 1), random.uniform(0, 1)),
            displacement_phi=(random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)),
            kerr=(random.uniform(-1, 1), random.uniform(-1, 1)),
        )
        for _ in range(num_layers)
    ]
    return FraudDetectionHybridModel(layers)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybridModel",
    "FraudDetectionHybridModelFactory",
]
