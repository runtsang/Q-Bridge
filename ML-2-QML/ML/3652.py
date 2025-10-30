from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSelfAttention(nn.Module):
    """
    Classical hybrid self‑attention that mimics the quantum interface.
    The module first applies a 2×2 convolution (as in a quanvolution filter)
    to produce a set of feature maps and then runs a standard scaled‑dot‑product
    self‑attention over the flattened patches.
    The API accepts `rotation_params` and `entangle_params` so it can be
    swapped with its quantum counterpart without changing the surrounding code.
    """

    def __init__(self, embed_dim: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(1, embed_dim, kernel_size=kernel_size, stride=stride)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs
            Tensor of shape (B, 1, H, W) containing grayscale images.
        rotation_params, entangle_params
            Optional placeholders to keep the signature compatible with the
            quantum implementation. They are ignored in the classical
            implementation but can be used to modulate the linear layers if
            desired.
        Returns
        -------
        output
            Tensor of shape (B, embed_dim) after attention aggregation.
        """
        # Feature extraction via 2×2 convolution (quanvolution style)
        x = self.conv(inputs)  # (B, E, H', W')
        B, E, H, W = x.shape
        # Flatten spatial dimensions
        x = x.view(B, E, -1)  # (B, E, N)
        # Project to Q, K, V
        Q = self.q_proj(x)  # (B, E, N)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # Scaled dot‑product attention
        attn_scores = torch.softmax((Q.transpose(1, 2) @ K) / np.sqrt(self.embed_dim), dim=-1)
        out = V @ attn_scores.transpose(1, 2)  # (B, E, N)
        # Aggregate over patches
        return out.mean(dim=-1)  # (B, E)

__all__ = ["HybridSelfAttention"]
