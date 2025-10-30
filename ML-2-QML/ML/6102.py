"""Enhanced classical quanvolution network with multi‑head self‑attention.

The network extracts 2×2 patches via a convolution, aggregates the patch
embeddings using a multi‑head self‑attention module, and classifies the
result with a linear head.  The architecture is fully trainable with
standard PyTorch optimisers and supports end‑to‑end training on
MNIST‑like data.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Quanvolution(nn.Module):
    """Classical quanvolution block with attention.

    Parameters
    ----------
    patch_size : int, default 2
        Size of the image patch to extract.
    out_channels : int, default 4
        Number of channels produced by the patch extractor.
    stride : int, default 2
        Stride of the patch extractor.
    num_heads : int, default 4
        Number of attention heads.
    d_k : int, default 16
        Dimension of the key/query/value vectors per head.
    """

    def __init__(
        self,
        patch_size: int = 2,
        out_channels: int = 4,
        stride: int = 2,
        num_heads: int = 4,
        d_k: int = 16,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=stride,
        )
        # The embedding dimension for the attention module is out_channels.
        self.attention = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            batch_first=True,
        )
        # Compute number of patches per image
        self.num_patches = (28 // patch_size) ** 2
        self.classifier = nn.Linear(out_channels * self.num_patches, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (B, 10).
        """
        # Extract patches
        features = self.conv(x)  # (B, out_channels, H', W')
        B, C, H, W = features.shape
        # Flatten spatial dimensions
        seq = features.reshape(B, C, H * W).transpose(1, 2)  # (B, seq_len, C)
        # Self‑attention
        attn_out, _ = self.attention(seq, seq, seq)  # (B, seq_len, C)
        # Flatten again for classification
        attn_flat = attn_out.reshape(B, -1)  # (B, C * seq_len)
        logits = self.classifier(attn_flat)
        return F.log_softmax(logits, dim=-1)
