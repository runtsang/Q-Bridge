from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalSelfAttention:
    """
    Classical self‑attention that mimics the quantum interface.  The two
    parameter matrices are trainable tensors that emulate rotation and
    entanglement gates.
    """
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        q_mat = torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        k_mat = torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)

        query = torch.as_tensor(inputs @ q_mat, dtype=torch.float32)
        key   = torch.as_tensor(inputs @ k_mat, dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)

        scores = torch.softmax(query @ key.transpose(-1, -2) / math.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

def SelfAttention():
    """Return a ClassicalSelfAttention instance with the default embedding size."""
    return ClassicalSelfAttention(embed_dim=4)

class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        scores = self.dropout(F.softmax(scores, dim=-1))
        attn = torch.matmul(scores, v)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(attn)

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class UnifiedSelfAttentionTransformer(nn.Module):
    """
    Hybrid model that combines a classical self‑attention layer with a
    stack of classical transformer blocks.  The self‑attention parameters
    are trainable tensors that mirror the rotation/entanglement matrices
    used in the quantum reference.  The transformer part is fully
    classical but follows the same API as its quantum‑enhanced
    counterpart, enabling side‑by‑side comparison.
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
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)

        # Trainable parameters that emulate quantum rotation/entanglement
        self.rotation_params = nn.Parameter(torch.randn(3 * embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim - 1))

        self.self_attention = ClassicalSelfAttention(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        if num_classes > 2:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)

        # Classical self‑attention
        x_np = x.detach().cpu().numpy()
        sa_out = self.self_attention.run(
            self.rotation_params.detach().cpu().numpy(),
            self.entangle_params.detach().cpu().numpy(),
            x_np,
        )
        x = torch.from_numpy(sa_out).to(x.device)

        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "ClassicalSelfAttention",
    "SelfAttention",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "UnifiedSelfAttentionTransformer",
]
