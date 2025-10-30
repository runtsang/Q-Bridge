"""Hybrid classical self‑attention model for fraud detection.

This module builds on the classical SelfAttention and FraudDetection
implementations.  It exposes a single class ``FraudSelfAttention`` that
combines a lightweight self‑attention block with a stack of
photonic‑style fully‑connected layers.  The interface mirrors the
original ``SelfAttention`` factory function so that existing code
continues to run unchanged.

Key design points
-----------------
*  The attention block is a standard query‑key‑value network that
   accepts externally supplied rotation and entanglement parameters.
   This keeps the interface compatible with the quantum counterpart
   while remaining fully classical.
*  Fraud detection layers are instantiated from ``FraudLayerParameters``
   and are identical to the photonic implementation – linear
   transformations followed by tanh, scaling, and shifting.
*  The final output is a single‑dimensional regression value.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Parameter container – identical to the photonic version
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

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Build a single fraud‑detection layer."""
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
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

# --------------------------------------------------------------------------- #
# Self‑attention block
# --------------------------------------------------------------------------- #
class _SelfAttentionBlock(nn.Module):
    """Query‑key‑value attention that uses externally supplied parameters."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
    ) -> torch.Tensor:
        # rotation_params shape: (embed_dim * 3,)
        # entangle_params shape: (embed_dim - 1,)
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key = inputs @ entangle_params.reshape(self.embed_dim, -1)
        scores = torch.softmax(query @ key.T / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32)), dim=-1)
        context = scores @ inputs
        return context

# --------------------------------------------------------------------------- #
# Main hybrid model
# --------------------------------------------------------------------------- #
class FraudSelfAttention(nn.Module):
    """Hybrid classical self‑attention + fraud‑detection stack."""
    def __init__(self, embed_dim: int, fraud_params: Sequence[FraudLayerParameters]):
        super().__init__()
        self.attention = _SelfAttentionBlock(embed_dim)
        self.fraud_layers = nn.Sequential(*[_layer_from_params(p, clip=True) for p in fraud_params])
        self.final_linear = nn.Linear(2, 1)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
    ) -> torch.Tensor:
        ctx = self.attention(inputs, rotation_params, entangle_params)
        out = self.fraud_layers(ctx)
        out = self.final_linear(out)
        return out

# --------------------------------------------------------------------------- #
# Factory function matching the original interface
# --------------------------------------------------------------------------- #
def SelfAttention() -> FraudSelfAttention:
    # Dummy fraud parameters – in practice the user would supply real ones
    dummy_params = [
        FraudLayerParameters(
            bs_theta=0.5,
            bs_phi=0.5,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
    ]
    return FraudSelfAttention(embed_dim=4, fraud_params=dummy_params)

__all__ = ["SelfAttention", "FraudLayerParameters", "FraudSelfAttention"]
