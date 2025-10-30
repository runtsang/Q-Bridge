"""Combined classical fraud‑detection with self‑attention.

This module defines a PyTorch model that first applies a self‑attention
layer (adapted from the Qiskit example) and then passes the result through
a stack of fraud‑detection layers (inspired by the photonic seed).  The
model can be trained end‑to‑end and is fully compatible with standard
PyTorch training loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Parameter container – mirrors the photonic definition
# --------------------------------------------------------------------------- #
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
    """Utility to bound a float – used when constructing layers."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, clip: bool = False) -> nn.Module:
    """Instantiate a single fraud‑detection layer from parameters."""
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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


# --------------------------------------------------------------------------- #
# Classical self‑attention – adapted from the Qiskit example
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention:
    """Simple multi‑head self‑attention that operates on a 2‑D input."""

    def __init__(self, embed_dim: int = 4):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        # Project inputs to query/key/value spaces
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)

        # Compute attention scores and weighted sum
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        out = scores @ value
        return out.numpy()


# --------------------------------------------------------------------------- #
# Main model – combines attention and fraud layers
# --------------------------------------------------------------------------- #
class FraudDetectionAttentionModel(nn.Module):
    """Hybrid model that applies self‑attention followed by fraud‑detection layers."""

    def __init__(
        self,
        attention_params: tuple[np.ndarray, np.ndarray],
        fraud_params: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.attention = ClassicalSelfAttention()
        self.fraud_layers = nn.ModuleList(
            [_layer_from_params(p, clip=True) for p in fraud_params]
        )
        self.output = nn.Linear(2, 1)

        # Store parameters for reproducibility
        self.attention_params = attention_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self‑attention step – expects a NumPy array
        attn_out = self.attention.run(
            self.attention_params[0], self.attention_params[1], x.numpy()
        )
        # Convert back to tensor
        out = torch.as_tensor(attn_out, dtype=torch.float32)

        # Fraud‑detection layers
        for layer in self.fraud_layers:
            out = layer(out)

        return self.output(out)


__all__ = ["FraudLayerParameters", "FraudDetectionAttentionModel"]
