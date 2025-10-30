"""Hybrid fully‑connected layer combining linear mapping with patch extraction.

The module defines ``HybridFCL`` which can be used as a drop‑in replacement for
a standard fully‑connected layer.  When a 2‑D image tensor is supplied, the
layer extracts non‑overlapping 2×2 patches, flattens them, and processes the
resulting feature map with a linear head.  This merges the lightweight linear
mapping of the original FCL with the spatial awareness of the quanvolution
filter.
"""

import torch
import torch.nn as nn

class HybridFCL(nn.Module):
    """Fully‑connected layer with optional 2×2 patch extraction.

    Parameters
    ----------
    in_features : int
        Number of input features for the linear transformation.
    out_features : int
        Number of output features.
    patch_size : int | None, optional
        If provided, the layer will treat the input as a 2‑D image and
        apply a non‑overlapping patch extraction with this size.
    """

    def __init__(self, in_features: int, out_features: int, patch_size: int | None = None):
        super().__init__()
        self.patch_size = patch_size
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch extraction for 2‑D images
        if self.patch_size is not None and x.dim() == 4:
            b, c, h, w = x.shape
            assert h % self.patch_size == 0 and w % self.patch_size == 0, (
                f"Image dimensions {h}x{w} are not divisible by patch_size={self.patch_size}"
            )
            patches = (
                x.unfold(2, self.patch_size, self.patch_size)
                .unfold(3, self.patch_size, self.patch_size)
                .contiguous()
                .view(b, c, -1, self.patch_size * self.patch_size)
                .permute(0, 2, 1, 3)
                .reshape(b, -1, c * self.patch_size * self.patch_size)
            )
            return self.linear(patches)
        else:
            return self.linear(x.view(x.size(0), -1))

__all__ = ["HybridFCL"]
