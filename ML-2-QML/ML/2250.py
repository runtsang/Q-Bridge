from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """Depthwise separable convolution filter with optional weight sharing."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 depthwise: bool = True, shared_weights: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.depthwise = depthwise
        self.shared_weights = shared_weights

        if self.depthwise:
            self.depthwise_conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
            self.pointwise = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        else:
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> float:
        if data.ndim == 2:
            data = data.unsqueeze(0)
        if data.ndim == 3:
            data = data.unsqueeze(1)
        else:
            raise ValueError("Input must be 2‑D or 3‑D (batch).")

        if self.depthwise:
            out = self.depthwise_conv(data)
            out = self.pointwise(out)
        else:
            out = self.conv(data)

        logits = out
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, batch_size: int) -> torch.Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, self.attn_weights = self.attention(q, k, v)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented classically."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        return self.combine_heads(self.downstream(q, k, v, x.size(0)))

class FeedForwardBase(nn.Module):
    """Base feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

class TransformerBlockClassical(TransformerBlockBase):
    """Standard transformer block with classical sub‑modules."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

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
    """Transformer‑based text classifier supporting classical sub‑modules."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(*[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                                            for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

class ConvTransformerQuantum(nn.Module):
    """
    Hybrid model that first applies a convolutional filter to each input image
    and then processes the flattened token sequence with a transformer.
    This classical implementation uses the standard transformer sub‑modules.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 conv_depthwise: bool = True,
                 conv_shared_weights: bool = False,
                 vocab_size: int = 30522,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 num_blocks: int = 2,
                 ffn_dim: int = 256,
                 num_classes: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.conv_filter = ConvFilter(kernel_size, threshold,
                                      depthwise=conv_depthwise,
                                      shared_weights=conv_shared_weights)
        self.transformer = TextClassifier(vocab_size=vocab_size,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          num_blocks=num_blocks,
                                          ffn_dim=ffn_dim,
                                          num_classes=num_classes,
                                          dropout=dropout)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        images: (B, H, W) or (B, H, W, C) with C=1
        tokens: (B, T) integer token ids
        """
        conv_outputs = torch.tensor([self.conv_filter(img) for img in images], dtype=torch.float32)
        token_emb = self.transformer.token_embedding(tokens)
        conv_expanded = conv_outputs.unsqueeze(-1).expand_as(token_emb)
        x = token_emb + conv_expanded
        x = self.transformer.pos_embedding(x)
        x = self.transformer.transformers(x)
        x = self.transformer.dropout(x.mean(dim=1))
        return self.transformer.classifier(x)
