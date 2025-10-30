"""Hybrid quanvolution + transformer model for regression/classification.

The classical implementation builds on a 2‑pixel quanvolution filter, a sequence of
Transformer blocks, and a small regression head.  The API mirrors the quantum
implementation so that the same class name can be used interchangeably in
experiment scripts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Classical quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilterClassical(nn.Module):
    """Extract 2×2 patches from a 28×28 image using a 2‑channel convolution."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)


# --------------------------------------------------------------------------- #
#  Classical Transformer components
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return out


class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Hybrid model
# --------------------------------------------------------------------------- #
class HybridQuanvolutionTransformerEstimator(nn.Module):
    """Hybrid model mixing quanvolution, Transformers and a regression head.

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    out_channels : int
        Number of channels after the quanvolution filter.
    embed_dim : int
        Embedding dimension for transformer layers.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension in the feed‑forward sub‑network.
    regression_hidden : int
        Hidden size of the regression head.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 128,
        regression_hidden: int = 32,
    ) -> None:
        super().__init__()
        # Quanvolution filter
        self.filter = QuanvolutionFilterClassical(in_channels, out_channels)

        # Projection to transformer embedding
        self.proj = nn.Linear(out_channels * 14 * 14, embed_dim)

        # Transformer blocks
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)]
        )

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, regression_hidden),
            nn.ReLU(),
            nn.Linear(regression_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        feats = self.filter(x)                     # [B, N]
        seq = self.proj(feats).unsqueeze(1)        # [B, 1, D]
        seq = self.transformer(seq)                # [B, 1, D]
        out = self.regressor(seq.mean(dim=1))      # [B, 1]
        return out


# --------------------------------------------------------------------------- #
#  Utility: simple classical estimator
# --------------------------------------------------------------------------- #
def EstimatorQNN() -> nn.Module:
    """Return a lightweight feed‑forward regressor for compatibility."""
    class Estimator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    return Estimator()


__all__ = [
    "QuanvolutionFilterClassical",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "HybridQuanvolutionTransformerEstimator",
    "EstimatorQNN",
]
