"""Hybrid classical regression model combining quantum‑inspired random
encoders and fraud‑detection style fully‑connected blocks.

This module extends the original QuantumRegression.py by adding
parameterised layers inspired by the fraud‑detection example.  The
model is fully classical (NumPy/PyTorch) and can be trained with
standard optimisers.  The dataset generator and RegressionDataset
are kept for backward compatibility.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Iterable, Tuple

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return a synthetic regression dataset.

    The data is generated in the same way as the original quantum example
    (sin + cos mixture) but the function is kept for backward compatibility.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class RegressionDataset(Dataset):
    """TensorDataset that mirrors the quantum data structure.

    Each sample contains a ``states`` tensor (float32) and a scalar ``target``.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Fraud‑style layer definition
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters for a fully‑connected block with clipping and scaling."""
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
    """Construct a single fraud‑style block."""
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Return a sequential model of fraud‑style blocks."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# Hybrid regression model
# --------------------------------------------------------------------------- #

class QModel(nn.Module):
    """Classical regression model that mimics the quantum architecture."""
    def __init__(self, num_features: int, num_layers: int = 3):
        super().__init__()
        # Random linear pre‑layer to act as a feature encoder
        self.pre = nn.Sequential(
            nn.Linear(num_features, 2),
            nn.ReLU(),
        )
        # Build fraud‑style layers
        # Randomly initialise parameters for each layer
        def _rand_param() -> FraudLayerParameters:
            return FraudLayerParameters(
                bs_theta=np.random.randn(),
                bs_phi=np.random.randn(),
                phases=tuple(np.random.randn(2)),
                squeeze_r=tuple(np.random.randn(2)),
                squeeze_phi=tuple(np.random.randn(2)),
                displacement_r=tuple(np.random.randn(2)),
                displacement_phi=tuple(np.random.randn(2)),
                kerr=tuple(np.random.randn(2)),
            )
        input_params = _rand_param()
        layer_params = [_rand_param() for _ in range(num_layers - 1)]
        self.fraud_net = build_fraud_detection_program(input_params, layer_params)
        self.head = nn.Linear(2, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.pre(state_batch)
        x = self.fraud_net(x)
        return self.head(x).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data", "FraudLayerParameters"]
