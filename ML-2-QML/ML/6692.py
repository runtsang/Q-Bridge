import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional

def SelfAttention():
    """
    Classical self‑attention helper mirroring the quantum circuit interface.
    Returns a class with a run method that accepts rotation_params,
    entangle_params and inputs and produces the attention output.
    """
    class ClassicalSelfAttention:
        def __init__(self, embed_dim: int):
            self.embed_dim = embed_dim

        def run(self,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray,
                inputs: np.ndarray) -> np.ndarray:
            query = torch.as_tensor(
                inputs @ rotation_params.reshape(self.embed_dim, -1),
                dtype=torch.float32)
            key = torch.as_tensor(
                inputs @ entangle_params.reshape(self.embed_dim, -1),
                dtype=torch.float32)
            value = torch.as_tensor(inputs, dtype=torch.float32)
            scores = torch.softmax(query @ key.T / math.sqrt(self.embed_dim), dim=-1)
            return (scores @ value).numpy()

    return ClassicalSelfAttention(embed_dim=4)

class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding as used in the reference transformer.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) *
            (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class MultiHeadAttention(nn.Module):
    """
    Classical multi‑head attention identical to the reference.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q = self.q_linear(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out)

class FeedForward(nn.Module):
    """
    Classical two‑layer MLP.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """
    Classical transformer block.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class HybridSelfAttentionTransformer(nn.Module):
    """
    A transformer‑based text classifier that can be used as a drop‑in replacement
    for the original SelfAttention and TextClassifier modules.
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
        self.transformer_layers = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer_layers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)
