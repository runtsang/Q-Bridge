"""Hybrid classical–quantum transformer for scalable NLP experiments."""

from __future__ import annotations

import math
from typing import Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Helper: lightweight experiment logger
# --------------------------------------------------------------------------- #
class ExperimentTracker:
    """Collects statistics on a per‑block basis during training."""

    def __init__(self, name: str):
        self.name = name
        self.metrics: dict[str, list[float]] = {}

    def log(self, key: str, value: float) -> None:
        self.metrics.setdefault(key, []).append(value)

    def report(self) -> dict[str, float]:
        return {k: sum(v) / len(v) for k, v in self.metrics.items()}


# --------------------------------------------------------------------------- #
# Shared base classes
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for all attention variants."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        *,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, T, E) → (B, H, T, d_k)."""
        B, T, E = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of separate_heads."""
        B, H, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Classical attention
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention built on torch.nn.MultiheadAttention."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout.p,
            batch_first=True,
        )

    def forward(self, *_, **__):
        raise NotImplementedError  # pragma: no cover

    def _attn_forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return self.merge_heads(attn_out)


# --------------------------------------------------------------------------- #
# Hybrid attention – classical + quantum kernel
# --------------------------------------------------------------------------- #
class HybridAttention(MultiHeadAttentionBase):
    """Attention that re‑uses an existing Q‑module to produce a linear projection."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Quantum kernel to map input embeddings to d_k
        self.quantum_proj = tq.QuantumModule(
            tq.QuantumDevice(
                n_wires=8,
                bsz=1,  # will be re‑used with dynamic bsz
                device=torch.device("cpu")
            ),
            tq.QuantumCircuit(
                tq.RX(0.0, wires=[0]),
                tq.RX(0.0, wires=[wires[0]]),
                **kwargs
            )
        )

    # TODO: implement forward logic
