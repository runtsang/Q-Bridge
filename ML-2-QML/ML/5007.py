"""Hybrid kernel‑attention model combining classical and quantum‑inspired components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
from torch import nn


# --------------------------------------------------------------------------- #
# 1. Fraud‑detection inspired linear‑activation layers
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters for a single fraud‑detection layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]],
                          dtype=torch.float32)
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
            return out * self.scale + self.shift

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Construct a sequential fraud‑detection style network."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# 2. Self‑attention head
# --------------------------------------------------------------------------- #
def SelfAttention():
    class ClassicalSelfAttention:
        def __init__(self, embed_dim: int):
            self.embed_dim = embed_dim

        def run(
            self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
        ) -> np.ndarray:
            query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1),
                                    dtype=torch.float32)
            key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1),
                                  dtype=torch.float32)
            value = torch.as_tensor(inputs, dtype=torch.float32)
            scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
            return (scores @ value).numpy()

    return ClassicalSelfAttention(embed_dim=4)


# --------------------------------------------------------------------------- #
# 3. CNN backbone (Quantum‑NAT inspired)
# --------------------------------------------------------------------------- #
class QFCModel(nn.Module):
    """CNN + FC projection to four features."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)


# --------------------------------------------------------------------------- #
# 4. Hybrid kernel‑attention model
# --------------------------------------------------------------------------- #
class HybridKernelAttentionModel(nn.Module):
    """
    A hybrid model that:
        1. Extracts 4‑D features via a CNN.
        2. Applies fraud‑detection style layers.
        3. Runs a classical self‑attention head.
        4. Emits an RBF Gram matrix between the attention outputs.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.cnn = QFCModel()
        # Example fraud‑detection parameters (identity‑like for demo)
        example_params = FraudLayerParameters(
            bs_theta=0.0, bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(1.0, 1.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.0, 0.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        self.fraud = build_fraud_detection_program(example_params, [])
        self.attention = SelfAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # 1. CNN feature extraction
        feat = self.cnn(x)          # shape (N, 4)
        # 2. Fraud‑detection style transformation
        fraud_out = self.fraud(feat)  # shape (N, 1)
        # 3. Feed‑forward to 4‑D space again for attention
        fraud_out = fraud_out.repeat(1, 4)  # dummy mapping to 4‑D
        # 4. Self‑attention
        rotation = np.identity(4)
        entangle = np.identity(4)
        attn_out = self.attention.run(rotation, entangle, fraud_out.detach().numpy())
        # 5. RBF kernel matrix between attention outputs
        return self._kernel_matrix(attn_out, attn_out)

    def _kernel_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = a[:, None, :] - b[None, :, :]
        return np.exp(-self.gamma * np.sum(diff * diff, axis=-1))


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "SelfAttention",
    "QFCModel",
    "HybridKernelAttentionModel",
]
