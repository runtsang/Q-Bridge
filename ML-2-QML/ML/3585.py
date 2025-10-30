import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from FCL import FCL

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Placeholder quantum attention – same interface but could be replaced by a real quantum module."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        noise = torch.randn_like(attn_out) * 0.01
        return attn_out + noise

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantumDummy(FeedForwardBase):
    """A lightweight classical emulation of a quantum feed-forward block."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * 0.01
        return self.linear2(self.dropout(F.relu(self.linear1(x + noise))))

class PositionalEncoder(nn.Module):
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

class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 use_quantum_attention: bool = False,
                 use_quantum_ffn: bool = False) -> None:
        super().__init__()
        self.attn = (MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
                     if use_quantum_attention else
                     MultiHeadAttentionClassical(embed_dim, num_heads, dropout))
        self.ffn = (FeedForwardQuantumDummy(embed_dim, ffn_dim, dropout)
                     if use_quantum_ffn else
                     FeedForwardClassical(embed_dim, ffn_dim, dropout))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class QTransformerTorch__gen214(nn.Module):
    """Unified transformer that can mix classical and quantum‑style components."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum_attention: bool = False,
                 use_quantum_ffn: bool = False) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, ffn_dim,
                             dropout,
                             use_quantum_attention,
                             use_quantum_ffn)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim,
                                    num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling
        x = self.dropout(x)
        return self.classifier(x)

__all__ = ["MultiHeadAttentionBase",
           "MultiHeadAttentionClassical",
           "MultiHeadAttentionQuantum",
           "FeedForwardBase",
           "FeedForwardClassical",
           "FeedForwardQuantumDummy",
           "PositionalEncoder",
           "TransformerBlock",
           "QTransformerTorch__gen214"]
