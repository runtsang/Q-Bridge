from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Base attention
class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, _ = self.attention(x, x, x, key_padding_mask=mask)
        return out

class MultiHeadAttentionQuantum(MultiHeadAttentionClassical):
    """Alias for API compatibility (classical implementation)."""

# Feed‑forward base
class FeedForwardBase(nn.Module):
    """Shared interface for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class FeedForwardQuantum(FeedForwardClassical):
    """Alias for API compatibility (classical implementation)."""

# Transformer block
class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    """Standard transformer block with classical sub‑modules."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attention = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.feedforward = FeedForwardClassical(embed_dim, ffn_dim, dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.feedforward(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockQuantum(TransformerBlockClassical):
    """Alias for API compatibility (classical implementation)."""

# Positional encoding
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

# Transformer‑based classifier
class TextClassifier(nn.Module):
    """Transformer‑based text classifier."""
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_blocks: int, ffn_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        self.encoder = nn.Sequential(*[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_enc(tokens)
        x = self.encoder(x)
        x = self.dropout(x.mean(dim=1))
        return self.output_layer(x)

# Classical LSTM‑based classifier
class LSTMClassifier(nn.Module):
    """LSTM‑based text classifier."""
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.lstm_module = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(x)
        outputs, _ = self.lstm_module(embeds)
        last_hidden = outputs[:, -1, :]
        out = self.dropout(last_hidden)
        return self.output_layer(out)

# Hybrid classifier
class HybridTextClassifier(nn.Module):
    """Hybrid transformer / LSTM text classifier."""
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_blocks: int, ffn_dim: int, num_classes: int, backbone: str = 'transformer', lstm_hidden_dim: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone_type = backbone
        if backbone == 'transformer':
            self.backbone = TextClassifier(vocab_size, embed_dim, num_heads, num_blocks, ffn_dim, num_classes, dropout)
        elif backbone == 'lstm':
            if lstm_hidden_dim is None:
                raise ValueError('lstm_hidden_dim must be specified for lstm backbone')
            self.backbone = LSTMClassifier(vocab_size, embed_dim, lstm_hidden_dim, num_classes, dropout)
        else:
            raise ValueError('backbone must be either "transformer" or "lstm"')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

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
    "LSTMClassifier",
    "HybridTextClassifier",
]
