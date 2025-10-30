"""Hybrid classical self‑attention module with fraud‑style clipping and scaling.

The module exposes a ClassicalSelfAttention class that computes
multi‑head self‑attention using torch tensors, and a
build_fraud_attention_model function that builds a lightweight
PyTorch sequential model mimicking the photonic fraud‑detection
pipeline.  The two components are deliberately co‑located so that
the same set of parameters can drive both a purely classical
attention block and, in the quantum branch, a quantum‑augmented
attention circuit.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple


@dataclasses.dataclass
class FraudLayerParameters:
    """Parameters for a fraud‑style attention layer."""
    scale: float
    shift: float
    clip: float = 5.0


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class ClassicalSelfAttention(nn.Module):
    """Multi‑head self‑attention implemented with torch, supporting
    fraud‑style clipping and scaling.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        fraud_params: FraudLayerParameters | None = None,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.fraud_params = fraud_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)

        if self.fraud_params is not None:
            attn = attn * self.fraud_params.scale + self.fraud_params.shift
            attn = torch.clamp(attn, -self.fraud_params.clip, self.fraud_params.clip)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        return out


def build_fraud_attention_model(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    embed_dim: int = 64,
    num_heads: int = 4,
) -> nn.Sequential:
    """Create a sequential PyTorch model that stacks fraud‑style attention
    layers followed by a linear classification head.
    """
    modules: list[nn.Module] = [
        ClassicalSelfAttention(embed_dim, num_heads, fraud_params=input_params)
    ]
    modules.extend(
        ClassicalSelfAttention(embed_dim, num_heads, fraud_params=layer)
        for layer in layers
    )
    modules.append(nn.Linear(embed_dim, 1))
    return nn.Sequential(*modules)


def SelfAttention() -> ClassicalSelfAttention:
    """Convenience constructor used by the anchor path."""
    return ClassicalSelfAttention(embed_dim=64, num_heads=4)
