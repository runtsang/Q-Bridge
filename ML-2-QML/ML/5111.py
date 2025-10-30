"""Hybrid regression model combining classical and quantum-inspired components.

The model merges ideas from the original quantum regression, fraud detection,
QRNN, and EstimatorQNN examples.  It uses a random unitary layer to mimic a
quantum feature map, a fraud‑style linear transformation with scaling/shift,
and a small classical head.

The API follows the original `RegressionDataset` and `QModel` naming
convention but renames the model to `HybridRegressionModel` to avoid clashes
with the quantum version.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Tuple

# --------------------------------------------------------------------------- #
#  Dataset utilities (borrowed from the original quantum regression example)
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a toy regression dataset where the target is a noisy sinusoid of the
    sum of the input features.  The inputs are uniformly sampled in [-1, 1].
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns a dict with `states` and `target` keys."""

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
#  Fraud‑style linear block (adapted from the photonic fraud example)
# --------------------------------------------------------------------------- #
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


class FraudLayer(nn.Module):
    """
    A lightweight implementation of the photonic fraud‑layer that maps a 2‑D input
    to a 2‑D output using a fixed linear transform, a non‑linearity, and a
    trainable scale/shift.
    """

    def __init__(self, params: FraudLayerParameters, clip: bool = True):
        super().__init__()
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
        linear = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        self.linear = linear
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(torch.tensor(params.displacement_r, dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.activation(self.linear(x))
        return out * self.scale + self.shift


# --------------------------------------------------------------------------- #
#  Quantum‑inspired feature map (classical implementation of a random unitary)
# --------------------------------------------------------------------------- #
class QuantumInspiredLayer(nn.Module):
    """
    Applies a random unitary transformation to the input (augmented with a bias
    term).  The unitary is a learnable torch.nn.Parameter, allowing the layer to
    adapt during training.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        rand = torch.randn(in_dim + 1, out_dim)
        q, _ = torch.linalg.qr(rand)  # Haar‑random orthonormal matrix
        self.unitary = nn.Parameter(q, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.ones(x.shape[0], 1, device=x.device)
        x_aug = torch.cat([x, bias], dim=1)
        return torch.tanh(x_aug @ self.unitary)


# --------------------------------------------------------------------------- #
#  Overall hybrid model
# --------------------------------------------------------------------------- #
class HybridRegressionModel(nn.Module):
    """
    Combines a quantum‑inspired feature map, a fraud‑style linear block, and a small
    classical head.  The architecture is deliberately lightweight to keep training
    fast while still exposing the interaction between classical and quantum‑ish
    components.
    """

    def __init__(self, num_features: int, fraud_params: FraudLayerParameters):
        super().__init__()
        self.quantum_layer = QuantumInspiredLayer(num_features, 32)
        self.fraud_layer = FraudLayer(fraud_params, clip=True)
        self.head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.quantum_layer(x)
        x = self.fraud_layer(x)
        return self.head(x).squeeze(-1)


__all__ = [
    "RegressionDataset",
    "generate_superposition_data",
    "FraudLayerParameters",
    "FraudLayer",
    "QuantumInspiredLayer",
    "HybridRegressionModel",
]
