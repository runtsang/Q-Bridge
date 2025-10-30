# QuantumConvTransformer – Classical implementation using a 2×2 convolution followed by a stack of classical transformer blocks.

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Conv import Conv


class MultiHeadAttentionBase(nn.Module):
    'Base class for multi‑head attention.'

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
    'Standard multi‑head attention built on torch.nn.MultiheadAttention.'

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:  # pragma: no cover
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out


class FeedForwardBase(nn.Module):
    'Base class for the position‑wise feed‑forward network.'

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    'Two‑layer MLP with ReLU.'

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockBase(nn.Module):
    'Base transformer block containing attention and feed‑forward parts.'

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    'Standard transformer block using classical attention and feed‑forward.'

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
    'Sinusoidal positional encoding.'

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


class QuantumConvTransformer(nn.Module):
    'Hybrid transformer that prepends a 2×2 convolution to the input and then applies a stack of classical transformer blocks.'

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = Conv()
        self.proj = nn.Linear(1, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[
                TransformerBlockClassical(
                    embed_dim, num_heads, ffn_dim, dropout
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, height, width).  The height and width are
            interpreted as a 2‑D image; the convolution filter processes each image
            independently.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes).
        """
        batch_size = x.size(0)
        conv_outputs = []
        for i in range(batch_size):
            conv_outputs.append(self.conv.run(x[i].cpu().numpy()))
        conv_vec = torch.tensor(conv_outputs, device=x.device, dtype=torch.float32).unsqueeze(-1)
        projected = self.proj(conv_vec)  # (batch, 1, embed_dim)
        seq_len = x.size(1)  # reuse height as sequence length
        tokens = projected.repeat(1, seq_len, 1)  # (batch, seq_len, embed_dim)
        x = self.pos_encoder(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)
