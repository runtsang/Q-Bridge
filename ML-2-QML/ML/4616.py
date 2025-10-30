import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockClassical(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads,
                                                dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class CNNFeatureExtractor(nn.Module):
    """2‑D CNN that produces a sequence of tokens for the transformer."""
    def __init__(self,
                 in_channels: int = 1,
                 num_tokens: int = 64,
                 embed_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        # 64 × 7 × 7 = 3136 feature map size after two 2×2 pools on 28×28 inputs
        self.token_proj = nn.Linear(64 * 7 * 7, num_tokens * embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        feat = self.conv(x)
        flat = self.flatten(feat)
        token_space = self.token_proj(flat)  # (bsz, num_tokens * embed_dim)
        tokens = token_space.view(bsz, -1, self.token_proj.out_features // token_space.size(1))
        return tokens


class HybridTransformerCNN(nn.Module):
    """
    Classical hybrid model combining CNN feature extraction and a transformer encoder.
    Parameters are chosen to mirror the quantum‑enhanced architecture in the reference.
    """
    def __init__(self,
                 num_classes: int,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_blocks: int = 4,
                 ffn_dim: int = 256,
                 num_tokens: int = 64,
                 dropout: float = 0.1,
                 in_channels: int = 1):
        super().__init__()
        self.cnn = CNNFeatureExtractor(in_channels, num_tokens, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=num_tokens)
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim,
                                      dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.cnn(x)                     # (bsz, num_tokens, embed_dim)
        tokens = self.pos_encoder(tokens)
        tokens = self.transformer(tokens)
        pooled = tokens.mean(dim=1)              # (bsz, embed_dim)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


__all__ = ["HybridTransformerCNN"]
