import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class HybridSelfAttention(nn.Module):
    """Classical self‑attention that mimics the original SelfAttention API."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray,
                inputs: np.ndarray) -> np.ndarray:
        W_q = torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        W_k = torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        Q = torch.as_tensor(inputs @ W_q, dtype=torch.float32)
        K = torch.as_tensor(inputs @ W_k, dtype=torch.float32)
        V = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(Q @ K.T / math.sqrt(self.embed_dim), dim=-1)
        return (scores @ V).numpy()

def SelfAttention():
    """Return an instance of the classical self‑attention class."""
    return HybridSelfAttention(embed_dim=4)

class MultiHeadAttention(nn.Module):
    """Standard multi‑head attention with batch‑first interface."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return out

class FeedForward(nn.Module):
    """Two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer block with residual connections."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, :x.size(1)]

class TextClassifier(nn.Module):
    """Transformer‑based text classifier."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor):
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = [
    "HybridSelfAttention",
    "SelfAttention",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "PositionalEncoder",
    "TextClassifier",
]
