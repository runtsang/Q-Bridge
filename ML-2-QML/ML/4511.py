from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Classical transformer components (adapted from QTransformerTorch)
class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(out)

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)

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
        return x + self.pe[:, :x.size(1)]

# Classical quanvolution filter (anchor)
class QuanvolutionFilter(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class QuanvolutionHybrid(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 filter_out_channels: int = 4,
                 num_transformer_blocks: int = 2,
                 embed_dim: int = 64,
                 num_heads: int = 4,
                 ffn_dim: int = 128,
                 num_classes: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        self.filter = QuanvolutionFilter(in_channels, filter_out_channels)
        self.sequence_len = 14 * 14  # 28x28 input with stride 2
        self.proj = nn.Linear(filter_out_channels * self.sequence_len, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_transformer_blocks)]
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)  # (batch, filter_out_channels * 196)
        seq = features.view(x.size(0), self.sequence_len, self.filter.out_channels)
        seq = self.proj(seq)
        seq = self.pos_encoder(seq)
        seq = self.transformer(seq)
        pooled = seq.mean(dim=1)
        logits = self.classifier(pooled)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
