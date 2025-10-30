"""Hybrid text classifier with a classical convolutional front‑end and
classical transformer blocks.

The class mirrors the API of the original QTransformerTorch
classifier but adds a configurable convolutional filter that
extracts patch activations from 2‑D inputs.  The architecture
is fully PyTorch‑based, making it drop‑in compatible with
existing training pipelines.

Author: gpt-oss-20b
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Classical convolutional filter
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Pure‑PyTorch 2‑D convolutional filter that outputs a scalar per patch.

    The filter consists of a single 2‑D convolution layer followed by a
    sigmoid activation.  It is designed to be a drop‑in replacement for
    the quantum quanvolution in the original project.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (B, H, W) or (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Patch activations of shape (B, 1, N_patches).
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)

        # Unfold into non‑overlapping patches
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
            padding=0,
        )  # (B, kernel_size*kernel_size, N_patches)

        # Compute logits and activations
        logits = self.conv(x).view(x.size(0), 1, -1, self.kernel_size, self.kernel_size)
        logits = logits.mean(dim=[3, 4])  # (B, 1, N_patches)
        activations = torch.sigmoid(logits - self.threshold)
        return activations


# --------------------------------------------------------------------------- #
# Classical transformer components
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockClassical(nn.Module):
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
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim),
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
# Hybrid classifier
# --------------------------------------------------------------------------- #
class HybridTextClassifier(nn.Module):
    """Transformer‑based classifier with an optional classical convolutional front‑end.

    Parameters
    ----------
    image_size : int
        Height/width of the square input image.
    patch_size : int
        Size of the square patches extracted by the convolutional filter.
    embed_dim : int
        Dimensionality of the transformer embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Feed‑forward network hidden dimension.
    num_classes : int
        Number of target classes.
    dropout : float, optional
        Drop‑out probability.
    use_quantum_conv : bool, optional
        Flag to switch to the quantum convolutional filter (ignored in this
        classical implementation).
    """
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum_conv: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2

        # Convolutional front‑end
        self.conv_filter = ConvFilter(kernel_size=patch_size)

        # Linear projection from scalar patch activation to embedding space
        self.patch_embed = nn.Linear(1, embed_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoder(embed_dim)

        # Transformer encoder
        self.transformer = nn.Sequential(
            *[
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim,
            num_classes if num_classes > 2 else 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, H, W) or (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes) or (B, 1) for binary.
        """
        # Convolutional filtering → (B, 1, N_patches)
        patch_activations = self.conv_filter(x)  # (B, 1, N_patches)

        # Project to embedding space
        x = self.patch_embed(patch_activations)  # (B, N_patches, embed_dim)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer(x)

        # Pool and classify
        x = x.mean(dim=1)  # global average pooling
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "ConvFilter",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "HybridTextClassifier",
]
