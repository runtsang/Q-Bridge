"""Enhanced multi‑head self‑attention module with dropout and positional encoding."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionML(nn.Module):
    """Classical self‑attention with multi‑head, dropout and positional encoding."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_lin = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_emb", self._sinusoidal_positional_encoding(1, embed_dim))

    def _sinusoidal_positional_encoding(self, seq_len: int, dim: int):
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * -(torch.log(torch.tensor(10000.0)) / dim)
        )
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, embed_dim]
        """
        seq_len = x.size(1)
        if self.pos_emb.size(0) < seq_len:
            self.pos_emb = self._sinusoidal_positional_encoding(seq_len, self.embed_dim)
        x = x + self.pos_emb[:seq_len]

        q = self.q_lin(x).reshape(x.size(0), seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_lin(x).reshape(x.size(0), seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_lin(x).reshape(x.size(0), seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).reshape(
            x.size(0), seq_len, self.embed_dim
        )
        output = self.out_lin(attn_output)
        return output

    def attention_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the averaged attention weights over heads for inspection.
        """
        seq_len = x.size(1)
        x = x + self.pos_emb[:seq_len]
        q = self.q_lin(x).reshape(x.size(0), seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_lin(x).reshape(x.size(0), seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return attn_weights.mean(1)  # average over heads


__all__ = ["SelfAttentionML"]
