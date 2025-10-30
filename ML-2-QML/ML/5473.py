"""Hybrid fraud detection model with classical transformer and photonic feature extractor.

This module builds on the original photonic fraud detection circuit by wrapping it in a
classical transformer.  The photonic layer acts as a feature extractor that feeds
the resulting 2‑dimensional representation into a stack of Transformer blocks.
The final classification head is a lightweight hybrid layer that mimics the
quantum expectation used in the QML counterpart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Photonic feature extractor (classical analogue)
# --------------------------------------------------------------------------- #

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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
#  Classical transformer components
# --------------------------------------------------------------------------- #

class ClassicalSelfAttention(nn.Module):
    """Self‑attention block that operates on a 2‑dimensional feature vector."""
    def __init__(self, embed_dim: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.proj(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(-1, self.embed_dim, 1) for t in qkv]
        scores = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(self.embed_dim)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v).reshape(-1, self.embed_dim)

class FeedForwardClassical(nn.Module):
    """Two‑layer MLP used in the transformer."""
    def __init__(self, embed_dim: int, ffn_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlockClassical(nn.Module):
    """Standard transformer block built from the classical attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int = 1, ffn_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# --------------------------------------------------------------------------- #
#  Hybrid classification head
# --------------------------------------------------------------------------- #

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics the quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Simple dense head that replaces the quantum circuit in the original model."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

# --------------------------------------------------------------------------- #
#  Main hybrid fraud detection model
# --------------------------------------------------------------------------- #

class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud detection model that fuses photonic feature extraction,
    a classical transformer stack, and a lightweight hybrid classification head.
    """
    def __init__(
        self,
        embed_dim: int = 2,
        num_heads: int = 1,
        num_blocks: int = 2,
        ffn_dim: int = 64,
        shift: float = 0.0,
        feature_layers: Sequence[FraudLayerParameters] | None = None,
        input_params: FraudLayerParameters | None = None,
    ) -> None:
        super().__init__()
        if input_params is None:
            raise ValueError("input_params must be provided")
        # Feature extractor
        self.feature_extractor = build_fraud_detection_program(
            input_params, feature_layers or []
        )
        # Transformer stack
        self.transformer = nn.Sequential(
            *[
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim)
                for _ in range(num_blocks)
            ]
        )
        self.pos_encoder = PositionalEncoder(embed_dim)
        # Hybrid head
        self.hybrid = Hybrid(1, shift=shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        feats = self.feature_extractor(inputs)
        # Transformer expects sequence dimension; add singleton seq dim
        seq = feats.unsqueeze(1)
        seq = self.pos_encoder(seq)
        seq = self.transformer(seq)
        seq = seq.squeeze(1)
        # Hybrid classification
        probs = self.hybrid(seq)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["FraudDetectionHybrid"]
