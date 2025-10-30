"""Hybrid self‑attention module combining classical attention with a convolutional feature extractor.

The class is fully differentiable and can be inserted into any PyTorch model. It
inherits the 2×2 patch extraction strategy from the quanvolution example and
applies a multi‑head self‑attention mechanism on the resulting embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionHybrid(nn.Module):
    """
    A hybrid self‑attention block that first extracts local features with a
    lightweight 2‑D convolution (inspired by the quanvolution filter) and then
    applies a multi‑head self‑attention mechanism.  The class is fully
    differentiable and can be inserted into any PyTorch model.
    """

    def __init__(self, embed_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Convolutional front‑end that mimics the 2×2 patch extraction of Quanvolution
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=2, stride=2, bias=False)
        # Linear projections for queries, keys and values
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, 1, H, W).  The module expects a single‑channel
            image such that a 2×2 patch extraction yields 14×14 patches for a 28×28
            image, exactly like the original quanvolution example.
        Returns
        -------
        torch.Tensor
            Output of shape (batch, embed_dim) – the aggregated attention vector.
        """
        # Patch embedding
        patches = self.patch_embed(x)          # (B, E, H/2, W/2)
        B, E, H, W = patches.shape
        patches = patches.view(B, E, -1).transpose(1, 2)  # (B, N, E)

        # Self‑attention
        q = self.q_proj(patches)
        k = self.k_proj(patches)
        v = self.v_proj(patches)

        attn_scores = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        context = attn_scores @ v
        out = self.out_proj(context.mean(dim=1))   # aggregate over tokens

        return out

__all__ = ["SelfAttentionHybrid"]
