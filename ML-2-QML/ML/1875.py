import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionBase(nn.Module):
    """Base class for multi-head attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, L, E) -> (B, H, L, D)."""
        B, L, E = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        self.attn_weights = scores
        return torch.matmul(scores, v)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi-head attention implemented in pure PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, **kwargs) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.kproj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.qproj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vproj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, E = x.shape
        k = self.kproj(x)
        q = self.qproj(x)
        v = self.vproj(x)
        k = self.separate_heads(k)
        q = self.separate_heads(q)
        v = self.separate_heads(v)
        attn_out = self.attention(q, k, v, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, E)
        return self.out_proj(attn_out)

class MultiHeadAttentionQuantum(MultiHeadAttentionClassical):
    """Placeholder for quantum-aware attention; implemented in QML module."""
    pass

class FeedForwardBase(nn.Module):
    """Base class for feed-forward subblocks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """Standard two-layer feed-forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int,
                 dropout: float = 0.1, **kwargs) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(FeedForwardClassical):
    """Placeholder for quantum feed-forward; implemented in QML module."""
    pass

class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed-forward."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    """Classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, **kwargs) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockQuantum(TransformerBlockClassical):
    """Placeholder for quantum transformer block; implemented in QML module."""
    pass

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
        return x + self.pe[:, :x.size(1)]

class TextClassifier(nn.Module):
    """Transformer-based text classifier."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.layers = nn.ModuleList(
            [TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
             for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

    def get_attention_weights(self):
        """Collect attention weights from all layers (if available)."""
        return [layer.attn.attn_weights for layer in self.layers if hasattr(layer.attn, "attn_weights")]

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifier",
]
