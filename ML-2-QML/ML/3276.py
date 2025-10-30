"""Classical transformer with self‑attention and optional quantum‑style interface.

The module intentionally keeps the classical core while exposing a helper that
mirrors the signature of the quantum implementation.  This allows downstream
experiments to swap between a purely classical transformer and a hybrid
quantum‑classical variant without changing the surrounding training code.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def classical_self_attention(
    rotation_params: np.ndarray,
    entangle_params: np.ndarray,
    inputs: np.ndarray,
) -> np.ndarray:
    """
    Minimal self‑attention routine that accepts the same arguments as the
    quantum implementation.  The function is pure NumPy/Torch and can be
    used as a drop‑in replacement when a quantum backend is not available.
    """
    embed_dim = inputs.shape[-1]
    # Reshape parameters to match linear projections
    q_w = rotation_params.reshape(embed_dim, -1)
    k_w = entangle_params.reshape(embed_dim, -1)
    # Linear projections
    q = torch.from_numpy(inputs @ q_w.T).float()
    k = torch.from_numpy(inputs @ k_w.T).float()
    v = torch.from_numpy(inputs).float()
    # Scaled dot‑product
    scores = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(embed_dim), dim=-1)
    out = scores @ v
    return out.detach().cpu().numpy()


class MultiHeadAttention(nn.Module):
    """Standard multi‑head attention implemented with linear layers."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq, _ = x.size()
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Two‑layer feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single transformer block with residual connections and layer‑norm."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-np.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class UnifiedSelfAttentionTransformer(nn.Module):
    """
    Classic transformer that can act as a drop‑in replacement for the quantum
    variant.  The architecture mirrors the quantum implementation so that the
    same downstream training loop can be reused.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_blocks: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)
