"""Hybrid self‑attention that merges classical transformer blocks with a
fully‑connected layer.  It preserves the ``SelfAttention`` API
(``run(rotation_params, entangle_params, inputs)``) from the original seed
while internally performing a multi‑head attention followed by a two‑layer
feed‑forward network.  The ``rotation_params`` and ``entangle_params``
arguments are retained for compatibility but are ignored in the classical
implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented with fully‑connected layers."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q = self.q_linear(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out)


class FeedForwardClassical(nn.Module):
    """Two‑layer feed‑forward network used after the attention block."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class HybridSelfAttention(nn.Module):
    """
    Classic transformer‑style self‑attention block with a feed‑forward sub‑network.
    The ``run`` method keeps the original signature from the seed module.
    """

    def __init__(self, embed_dim: int = 4, num_heads: int = 2,
                 ffn_dim: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

    def run(self, rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Preserve the original API.  ``rotation_params`` and ``entangle_params``
        are unused in the classical implementation; they are accepted only for
        backward compatibility.
        """
        x = torch.as_tensor(inputs, dtype=torch.float32).unsqueeze(0)  # add batch dim
        out = self.forward(x)
        return out.squeeze(0).detach().numpy()


def SelfAttention() -> HybridSelfAttention:
    """
    Compatibility wrapper that returns a pre‑configured instance.
    """
    return HybridSelfAttention(embed_dim=4, num_heads=2, ffn_dim=8, dropout=0.1)


__all__ = ["SelfAttention", "HybridSelfAttention"]
