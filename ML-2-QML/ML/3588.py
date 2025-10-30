from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------- Transformer primitives ---------------------------------- #
class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        q = self.q_linear(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(out)


class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


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


# ---------------------------------- QCNN + Transformer model ---------------------------------- #
class QCNNTransformerModel(nn.Module):
    """
    Classical QCNN model extended with transformer encoder blocks.
    The model first applies convolution‑like fully connected layers,
    then a stack of transformer blocks to capture long‑range dependencies.
    """
    def __init__(
        self,
        n_features: int = 8,
        conv_hidden: int = 16,
        n_conv_blocks: int = 3,
        n_transformer_blocks: int = 2,
        embed_dim: int = 8,
        num_heads: int = 2,
        ffn_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Convolution‑style feature extractor
        conv_layers = []
        in_ch = n_features
        for _ in range(n_conv_blocks):
            conv_layers.append(nn.Linear(in_ch, conv_hidden))
            conv_layers.append(nn.Tanh())
            in_ch = conv_hidden
        self.conv_extractor = nn.Sequential(*conv_layers)

        # Transformer encoder
        self.pos_enc = PositionalEncoder(embed_dim)
        self.transformers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(n_transformer_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input shape : (batch, features)"""
        x = self.conv_extractor(x)
        # project to embedding dimension if needed
        if x.size(-1)!= self.pos_enc.pe.size(-1):
            x = nn.functional.linear(x, torch.eye(self.pos_enc.pe.size(-1), x.size(-1)).to(x.device))
        x = self.pos_enc(x)
        for block in self.transformers:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return torch.sigmoid(self.classifier(x))


def QCNNTransformer() -> QCNNTransformerModel:
    """Factory returning a pre‑configured QCNNTransformerModel."""
    return QCNNTransformerModel()
