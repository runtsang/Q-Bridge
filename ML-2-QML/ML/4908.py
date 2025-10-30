"""Hybrid classical regression model with self‑attention and clipped fully connected layers.

The code mirrors the quantum counterpart (``HybridRegressionModel`` below) but is
fully classical.  It is compatible with the original ``QuantumRegression.py``
anchor by providing identical public APIs and data loaders.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Data generation – same distribution as the quantum example
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate data in the form ``x`` → ``y = sin(∑x) + 0.1*cos(2∑x)``."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class HybridRegressionDataset(Dataset):
    """Dataset that returns features and labels for regression."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Auxiliary components – inspired by FraudDetection and SelfAttention
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters that control a clipped linear layer."""
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
    input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# Classical self‑attention – inspired by the quantum version
# --------------------------------------------------------------------------- #
def ClassicalSelfAttention(embed_dim: int = 4) -> nn.Module:
    class _SelfAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_dim = embed_dim

        def forward(
            self,
            rotation_params: torch.Tensor,
            entangle_params: torch.Tensor,
            inputs: torch.Tensor,
        ) -> torch.Tensor:
            query = inputs @ rotation_params.reshape(self.embed_dim, -1)
            key = inputs @ entangle_params.reshape(self.embed_dim, -1)
            value = inputs
            scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
            return scores @ value

    return _SelfAttention()

# --------------------------------------------------------------------------- #
# Fully connected quantum layer – classical surrogate
# --------------------------------------------------------------------------- #
def FullyConnectedLayer(n_features: int = 1) -> nn.Module:
    class _FCL(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def forward(self, thetas: Iterable[float]) -> torch.Tensor:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation

    return _FCL()

# --------------------------------------------------------------------------- #
# Hybrid regression model – classical backbone
# --------------------------------------------------------------------------- #
class HybridRegressionModel(nn.Module):
    """
    A hybrid regression model that combines:

    1. a classical feed‑forward head,
    2. a self‑attention block,
    3. a fraud‑detection style clipped linear stack,
    4. a fully‑connected layer.

    The architecture is deliberately modular to facilitate ablation studies.
    """
    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 4,
        fraud_layers: Sequence[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, 32)
        self.attention = ClassicalSelfAttention(attention_dim)
        self.fraud_head = build_fraud_detection_program(
            FraudLayerParameters(
                bs_theta=0.5,
                bs_phi=0.5,
                phases=(0.1, 0.1),
                squeeze_r=(0.3, 0.3),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.2, 0.2),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            ),
            fraud_layers or [],
        )
        self.fcl = FullyConnectedLayer(2)
        self.output = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.encoder(x))
        attn_out = self.attention(
            torch.randn(self.attention.embed_dim, 2, device=x.device),
            torch.randn(self.attention.embed_dim, 2, device=x.device),
            x,
        )
        fraud_out = self.fraud_head(x)
        fcl_out = self.fcl(fraud_out.squeeze(-1).tolist())
        out = self.output(torch.cat([attn_out, fcl_out], dim=-1))
        return out.squeeze(-1)

__all__ = [
    "HybridRegressionModel",
    "HybridRegressionDataset",
    "generate_superposition_data",
    "build_fraud_detection_program",
    "ClassicalSelfAttention",
    "FullyConnectedLayer",
]
