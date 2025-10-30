"""
Module: ConvAttentionHybrid
Author: gpt-oss-20b
Description:
    A hybrid classical architecture that fuses a convolutional filter with a self‑attention
    mechanism. The Conv filter operates on 2‑D patches, producing a scalar per patch.
    These scalars are embedded into a fixed‑dimensional space and fed into a classical
    self‑attention block. The resulting attention output is projected to a single
    scalar per input sample.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

# Import the seed implementations
from Conv import Conv
from SelfAttention import SelfAttention


class ConvAttentionModule(nn.Module):
    """
    Classical convolution + self‑attention module.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolutional kernel (patch size).
    threshold : float, default 0.0
        Threshold used by the Conv filter.
    embed_dim : int, default 4
        Dimensionality of the patch embedding fed into attention.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, embed_dim: int = 4) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.embed_dim = embed_dim

        # Instantiate the seed Conv filter
        self.conv_filter = Conv()

        # Self‑attention block
        self.attention = SelfAttention()

        # Linear projection of attention output to a scalar
        self.proj = nn.Linear(self.embed_dim, 1)

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract non‑overlapping patches from the input image.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, 1, H, W).

        Returns
        -------
        patches : torch.Tensor
            Tensor of shape (B, Npatch, 1) where Npatch = (H‑k+1)*(W‑k+1).
        """
        B, C, H, W = x.shape
        patches = x.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        # patches shape (B, 1, H-k+1, W-k+1, k, k)
        patches = patches.contiguous().view(B, 1, -1, self.kernel_size, self.kernel_size)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, 1, H, W).

        Returns
        -------
        out : torch.Tensor
            Tensor of shape (B,) containing the scalar output per sample.
        """
        # 1. Patch extraction
        patches = self._extract_patches(x)  # (B, 1, Npatch, k, k)
        B, C, Npatch, k, _ = patches.shape

        # 2. Convolutional filtering per patch
        conv_out = []
        for idx in range(Npatch):
            patch = patches[:, :, idx, :, :].squeeze(1)  # (B, k, k)
            # Convert to numpy for the seed Conv filter
            patch_np = patch.cpu().numpy()
            # Apply Conv filter (returns scalar)
            scalar = self.conv_filter.run(patch_np)
            conv_out.append(np.full((B,), scalar, dtype=np.float32))
        conv_out = np.stack(conv_out, axis=1)  # (B, Npatch)

        # 3. Embed scalar into a fixed‑dimensional vector
        # Simple replication + linear transform
        embed = np.tile(conv_out[:, :, None], (1, 1, self.embed_dim))  # (B, Npatch, embed_dim)
        embed = embed.reshape(B, -1)  # (B, Npatch*embed_dim)

        # 4. Classical self‑attention
        # Generate dummy parameters for the seed implementation
        rotation_params = np.eye(self.embed_dim, dtype=np.float32).reshape(-1)
        entangle_params = np.eye(self.embed_dim, dtype=np.float32).reshape(-1)
        attn_out = self.attention.run(rotation_params, entangle_params, embed)  # (B, Npatch*embed_dim)

        # 5. Projection to scalar
        attn_tensor = torch.from_numpy(attn_out).float()
        out = self.proj(attn_tensor).squeeze(-1)  # (B,)
        return out


__all__ = ["ConvAttentionModule"]
