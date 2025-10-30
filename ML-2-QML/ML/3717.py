"""Hybrid transformer architecture with optional quantum‑inspired layers and fast evaluation utilities."""

from __future__ import annotations

import math
import numpy as np
from typing import Callable, List, Sequence, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Utility ----------
def _ensure_batch(values: Sequence[int]) -> torch.Tensor:
    """Convert a sequence of token indices to a batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.long)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


# ---------- Classical submodules ----------
class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented with torch.nn.MultiheadAttention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out


class FeedForwardClassical(nn.Module):
    """Two‑layer feed‑forward network with ReLU."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ---------- Quantum‑inspired simulation ----------
class QuantumSimulatedAttention(nn.Module):
    """
    A lightweight classical surrogate for quantum attention.
    It applies a parameter‑dependent rotation to each head before attention.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # rotation parameters per head
        self.rotations = nn.Parameter(torch.randn(num_heads))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        x_rot = x.view(batch, seq, self.num_heads, self.d_k)
        angles = self.rotations.view(1, 1, self.num_heads, 1)
        x_rot = torch.cos(angles) * x_rot + torch.sin(angles) * x_rot
        x_rot = x_rot.reshape(batch, seq, -1)
        return self.attn(x_rot, x_rot, x_rot, key_padding_mask=mask)[0]


class QuantumSimulatedFeedForward(nn.Module):
    """
    A simple quantum‑inspired feed‑forward module.
    It mixes a linear transformation with a sinusoidal non‑linearity mimicking a variational layer.
    """

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, ffn_dim)
        self.sine = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sine(self.dropout(F.relu(self.linear(x))))


# ---------- Transformer block ----------
class TransformerBlock(nn.Module):
    """
    A single transformer block that can operate in either classical or quantum‑simulated mode.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        if use_quantum:
            self.attn = QuantumSimulatedAttention(embed_dim, num_heads, dropout)
            self.ffn = QuantumSimulatedFeedForward(embed_dim, ffn_dim, dropout)
        else:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# ---------- Positional encoding ----------
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------- Hybrid transformer ----------
class HybridTransformer(nn.Module):
    """
    Transformer based text classifier that can seamlessly switch between classical and
    quantum‑inspired sub‑modules.  The API mirrors the original QTransformerTorch
    while exposing a lightweight `eval` interface inspired by FastBaseEstimator.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = PositionalEncoder(embed_dim)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    use_quantum=use_quantum,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    # ---------- Forward pass ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.positional(tokens)
        x = self.blocks(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

    # ---------- Evaluation utilities ----------
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[int]],
    ) -> List[List[float]]:
        """
        Fast evaluation of the model for a collection of input sequences.
        Parameters are token indices; observables are callables applied to the logits.
        """
        self.eval()
        observables = list(observables) or [lambda logits: logits.mean(dim=-1)]
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                batch = _ensure_batch(params)
                logits = self(batch)
                row: List[float] = []
                for obs in observables:
                    val = obs(logits)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(float(val))
                results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[int]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluates the model with Gaussian shot noise, mirroring FastEstimator.
        """
        base = self.evaluate(observables, parameter_sets)
        if shots is None:
            return base
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in base:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = [
    "HybridTransformer",
]
