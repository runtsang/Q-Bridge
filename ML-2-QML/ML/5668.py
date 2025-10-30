"""Hybrid self‑attention module that fuses a classical quanvolution filter with a vanilla attention block."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSelfAttentionFilter(nn.Module):
    """
    Classical hybrid module that first extracts local image patches with a 2×2 quanvolution
    (implemented as a simple Conv2d) and then applies a multi‑head self‑attention mechanism.
    The design mirrors the quantum interface: ``run`` accepts the same argument list
    but ignores the quantum‑specific parameters, making the API compatible with the
    quantum counterpart.
    """

    def __init__(self, embed_dim: int = 4, num_heads: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Quanvolution feature extractor – 2×2 patches, stride 2
        self.qconv = nn.Conv2d(1, embed_dim, kernel_size=2, stride=2)

        # Attention linear layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Attention‑weighted feature tensor of shape (batch, embed_dim).
        """
        # 1. Extract 2×2 patches
        patch_feats = self.qconv(x)          # (batch, embed_dim, 14, 14)
        patch_feats = patch_feats.view(x.size(0), -1)  # (batch, embed_dim*14*14)

        # 2. Project to query/key/value
        q = self.q_proj(patch_feats)         # (batch, embed_dim)
        k = self.k_proj(patch_feats)
        v = self.v_proj(patch_feats)

        # 3. Scaled dot‑product attention
        scores = torch.softmax((q @ k.t()) / np.sqrt(self.embed_dim), dim=-1)
        attn_output = scores @ v

        # 4. Final projection
        return self.out_proj(attn_output)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compatibility wrapper mimicking the quantum API.

        Parameters
        ----------
        rotation_params, entangle_params : np.ndarray
            Unused for the classical implementation but accepted for API consistency.
        inputs : np.ndarray
            Input image array of shape (batch, 1, 28, 28).

        Returns
        -------
        np.ndarray
            Classical attention output as a NumPy array.
        """
        torch_inputs = torch.as_tensor(inputs, dtype=torch.float32)
        out = self.forward(torch_inputs)
        return out.detach().cpu().numpy()


__all__ = ["HybridSelfAttentionFilter"]
