import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Classical building blocks
# ------------------------------------------------------------
class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention for the classical transformer."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)


class FeedForwardClassical(nn.Module):
    """Two‑layer MLP used after attention."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockClassical(nn.Module):
    """A single transformer encoder layer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ------------------------------------------------------------
# Optional preprocessing modules
# ------------------------------------------------------------
class ConvFilter(nn.Module):
    """A lightweight 2‑D convolution filter that mimics a quanvolution layer."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply conv followed by sigmoid thresholding."""
        x = self.conv(data)
        return torch.sigmoid(x - self.threshold)


class EstimatorNN(nn.Module):
    """Small fully‑connected network used for regression tasks."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


# ------------------------------------------------------------
# Hybrid transformer supporting optional quantum blocks
# ------------------------------------------------------------
class QTransformerHybrid(nn.Module):
    """Unified transformer that can operate purely classically or with quantum sub‑modules."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_conv: bool = False,
        use_regressor: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)

        # Build encoder stack
        self.blocks = nn.ModuleList(
            [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        # Optional modules
        self.use_conv = use_conv
        if use_conv:
            self.conv_filter = ConvFilter()
        else:
            self.conv_filter = None

        self.use_regressor = use_regressor
        if use_regressor:
            self.regressor = EstimatorNN()
        else:
            self.regressor = None

        # Final classifier
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor. For plain text classification it is a 2‑D tensor of word indices.
            If ``use_conv`` is True, ``x`` should be a 4‑D tensor representing a single‑channel
            image of shape (B, 1, H, W) that is first passed through the conv filter
            and flattened into token indices via a simple linear mapping.
        """
        # Conv preprocessing (optional)
        if self.use_conv:
            # Assumes input shape (B, 1, H, W) with H=W=kernel_size
            conv_out = self.conv_filter(x)  # (B, 1, 1, 1) -> (B, 1, 1, 1)
            # Convert conv output to token ids by simple threshold
            token_ids = conv_out.squeeze([1, 2, 3]).long()
            x = self.token_embedding(token_ids)
        else:
            x = self.token_embedding(x)

        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)

        x = self.dropout(x.mean(dim=1))  # global average pooling

        if self.use_regressor:
            # If a regressor is attached, feed the pooled representation to it
            x = self.regressor(x)
            return x

        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "ConvFilter",
    "EstimatorNN",
    "QTransformerHybrid",
]
