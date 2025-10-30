"""Hybrid fraud detection model with classical self‑attention and photonic layers.

The model follows the photonic architecture from the original seed but
pre‑processes the input with a learnable self‑attention block.
The attention parameters are trainable tensors, and the attention output
serves as the input to the photonic core.  All layers use clipping
to keep parameters in a physically meaningful range, mirroring the
photonic implementation.

Typical usage:
    model = FraudDetectionHybridAttention(input_params, layers)
    logits = model(inputs)   # inputs: (batch, 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
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


class ClassicalSelfAttention(nn.Module):
    """Learnable self‑attention block used in the fraud‑detection pipeline."""
    def __init__(self, embed_dim: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        # rotation and entangle parameters are trainable
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch, embed_dim)
        returns: (batch, embed_dim)
        """
        query = inputs @ self.rotation
        key = inputs @ self.entangle
        scores = torch.softmax(query @ key.transpose(-1, -2) / (self.embed_dim ** 0.5), dim=-1)
        return scores @ inputs


class FraudDetectionHybridAttention(nn.Module):
    """
    Hybrid fraud detection model that stacks a self‑attention block
    followed by the photonic‑style layers defined by ``FraudLayerParameters``.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim=2)
        # Build the photonic‑style sequential core
        modules = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 1))
        self.core = nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch, 2)
        returns: fraud probability logits (batch, 1)
        """
        # Self‑attention on the raw input
        attn_out = self.attention(inputs)
        # Feed the attention output into the photonic core
        return self.core(attn_out)

__all__ = ["FraudLayerParameters", "ClassicalSelfAttention", "FraudDetectionHybridAttention"]
