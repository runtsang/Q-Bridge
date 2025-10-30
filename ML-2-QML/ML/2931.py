"""Hybrid quanvolutional model combining classical convolution, quantum‑kernel emulation, and self‑attention.

This module defines :class:`QuanvolutionHybrid` which can replace the original
``Quanvolution.py``.  It is fully classical (NumPy/PyTorch) and contains three
main components:

* :class:`ClassicalQuanvolutionFilter` – a patch‑wise feature extractor that
  emulates a two‑qubit quantum kernel via a fixed random projection.
* :class:`ClassicalSelfAttention` – a standard multi‑head self‑attention block
  operating on the patch feature map.
* :class:`QuanvolutionHybrid` – a classifier that combines the two modules and
  feeds the attended features to a linear head.

The implementation is ready for integration into existing training pipelines.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QuanvolutionHybrid"]


class ClassicalQuanvolutionFilter(nn.Module):
    """Patch‑wise quantum‑kernel emulation using a fixed random projection.

    The filter splits a 28×28 image into non‑overlapping 2×2 patches, flattens each
    patch to a vector of length ``in_channels * 4`` and projects it to
    ``out_channels`` dimensions via a fixed random matrix.  No learnable
    parameters are introduced – the projection is a stand‑in for a random
    two‑qubit circuit.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4, patch_size: int = 2) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.projection = nn.Parameter(
            torch.randn(out_channels, in_channels * patch_size * patch_size), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        patches = (
            x.unfold(2, self.patch_size, self.patch_size)
           .unfold(3, self.patch_size, self.patch_size)
           .contiguous()
        )  # (B, C, n_h, n_w, patch_size, patch_size)
        n_patches = patches.shape[2] * patches.shape[3]
        patches = patches.view(B, C, -1, self.patch_size * self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patches = patches.view(B, n_patches, C * self.patch_size * self.patch_size)
        features = torch.einsum("bpc,oc->bpo", patches, self.projection)
        return features  # (B, n_patches, out_channels)


class ClassicalSelfAttention(nn.Module):
    """Standard multi‑head self‑attention block.

    The implementation follows the formulation in “Attention is All You Need” and
    is compatible with the output shape of :class:`ClassicalQuanvolutionFilter`.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, E)
        B, N, E = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, E // self.num_heads)
        q, k, v = qkv.unbind(dim=2)
        scores = torch.einsum("bnhd,bmhd->bnm", q, k) / np.sqrt(E / self.num_heads)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bnm,bmhd->bnhd", attn, v).reshape(B, N, E)
        return self.out_proj(out)


class QuanvolutionHybrid(nn.Module):
    """Hybrid quanvolutional classifier with quantum‑kernel emulation and self‑attention.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (default 1 for MNIST).
    out_channels : int
        Width of the feature map produced by :class:`ClassicalQuanvolutionFilter`.
    num_classes : int
        Number of target classes (default 10 for MNIST).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.filter = ClassicalQuanvolutionFilter(in_channels, out_channels)
        self.attention = ClassicalSelfAttention(embed_dim=out_channels)
        self.classifier = nn.Linear(out_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        features = self.filter(x)  # (B, 14*14, out_channels)
        attended = self.attention(features)  # (B, 14*14, out_channels)
        flat = attended.view(x.size(0), -1)  # (B, 14*14*out_channels)
        logits = self.classifier(flat)
        return F.log_softmax(logits, dim=-1)
