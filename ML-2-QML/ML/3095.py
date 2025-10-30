"""Hybrid fraud detection module combining photonic‑style layers with quantum self‑attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import numpy as np

# --------------------------------------------------------------------------- #
#  Photonic‑style layer definition (classical analogue)
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

# --------------------------------------------------------------------------- #
#  Classical self‑attention helper (mimicking the quantum interface)
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """Self‑attention block that accepts external rotation and entangle parameters."""

    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# --------------------------------------------------------------------------- #
#  Hybrid model
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid(nn.Module):
    """
    Combines a classical photonic‑style fraud‑detection backbone with a quantum‑style
    self‑attention mechanism.  The attention block is purely classical but
    follows the same API as the quantum variant, allowing seamless swapping.
    """

    def __init__(
        self,
        fraud_params: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        attention_params: tuple[np.ndarray, np.ndarray],
    ) -> None:
        super().__init__()
        self.backbone = build_fraud_detection_program(fraud_params, fraud_layers)
        self.attention = ClassicalSelfAttention(embed_dim=attention_params[0].shape[0])
        self.rotation_params, self.entangle_params = attention_params

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run a single inference through the hybrid network."""
        # Attention first – enhances raw features
        attn_out = self.attention(
            self.rotation_params, self.entangle_params, inputs
        )
        # Convert to torch tensor for the backbone
        x = torch.as_tensor(attn_out, dtype=torch.float32)
        out = self.backbone(x)
        return out.squeeze().detach().numpy()

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "ClassicalSelfAttention",
    "FraudDetectionHybrid",
]
