"""Hybrid Fraud‑Transformer model combining photonic fraud‑detection layers with a classical transformer encoder.

This module defines a single ``FraudTransformerQuantum`` class that
re‑uses the photonic‑style ``FraudLayerParameters`` from the original
``FraudDetection`` example and extends it with a transformer encoder.
The encoder can be instantiated either as a classic block (``TransformerBlockClassical``)
or as a quantum‑aware block (``TransformerBlockQuantum``).  The
quantum block in the quantum version supports optional quantum sub‑modules
(attention/FFN) in the quantum variant, but the classical version
uses standard PyTorch layers.

The design keeps the following traits:
* 2‑mode photonic layers with the same parameterization as before.
* A positional encoder that can be used on any embedding.
* Optional quantum sub‑modules for experiments.
* A clean interface that can be dropped into a research‑grade notebook
  or into a new ``FraudDetection__gen193.py`` file.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class FraudLayerParameters:
    """Container for the 2‑mode photonic layer parameters used in the
    original fraud‑detection model.  The same names are kept for
    compatibility with the existing seed code.
    """
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    """Clamp single float value into ``[-bound, bound]``."""
    return max(-bound, min(bound, value))

def _photonic_layer(params: FraudLayerParameters, *, clip: bool = False) -> nn.Module:
    """Return a small neural network that mimics a two‑mode photonic
    circuit.  The mapping is intentionally simple: a linear layer
    followed by a tanh, then a scale and shift.
    """
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

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class TransformerBlockClassical(nn.Module):
    """A simple transformer block with multi‑head attention and feed‑forward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class FraudTransformerQuantum(nn.Module):
    """Hybrid fraud‑detection model that combines photonic layers
    with a transformer encoder.  The transformer can be instantiated
    with either classical or quantum sub‑modules in the quantum
    variant (see the QML module).
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        num_blocks: int = 2,
        num_heads: int = 2,
        ffn_dim: int = 8,
        dropout: float = 0.1,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.input_params = input_params
        self.layers = list(layers)

        # Photonic encoder
        self.photonic = _photonic_layer(input_params, clip=False)

        # Positional encoding (works on 2‑dim embeddings)
        self.pos_encoder = PositionalEncoder(embed_dim=2)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlockClassical(2, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 2) representing raw fraud‑features.
        Returns:
            Tensor of shape (batch, num_classes) logits.
        """
        # Photonic layer
        out = self.photonic(x)  # (batch, 2)

        # Treat as a single‑token sequence for the transformer
        out = out.unsqueeze(1)  # (batch, 1, 2)
        out = self.pos_encoder(out)  # (batch, 1, 2)

        # Transformer stack
        for block in self.transformer_blocks:
            out = block(out)

        # Pooling across sequence length
        out = out.mean(dim=1)  # (batch, 2)
        out = self.dropout(out)
        return self.classifier(out)

def build_fraud_transformer(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *,
    num_blocks: int = 2,
    num_heads: int = 2,
    ffn_dim: int = 8,
    dropout: float = 0.1,
    num_classes: int = 1,
) -> FraudTransformerQuantum:
    """Return a fully‑configured ``FraudTransformerQuantum`` instance."""
    return FraudTransformerQuantum(
        input_params,
        layers,
        num_blocks=num_blocks,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        num_classes=num_classes,
    )

__all__ = [
    "FraudLayerParameters",
    "FraudTransformerQuantum",
    "build_fraud_transformer",
]
