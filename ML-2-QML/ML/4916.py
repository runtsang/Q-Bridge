"""Hybrid classical kernel combining RBF, self‑attention and patch‑based feature extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence

class HybridKernelMethod(nn.Module):
    """Hybrid kernel that combines a patchwise convolutional embedding,
    a classical self‑attention weighting, and an RBF similarity.
    This module is fully classical and can be used as a drop‑in replacement
    for the original ``Kernel`` class.

    Parameters
    ----------
    gamma : float, default 1.0
        RBF bandwidth.
    patch_size : int, default 2
        Size of the 2‑D patch to extract from each input image.
    attention : bool, default False
        Whether to apply a self‑attention weighting to the patch
        embeddings before computing the kernel.
    """

    def __init__(self, gamma: float = 1.0, patch_size: int = 2, attention: bool = False):
        super().__init__()
        self.gamma = gamma
        self.patch_size = patch_size
        self.attention = attention
        # simple 2‑D conv to extract patches
        self.conv = nn.Conv2d(1, 1, kernel_size=patch_size, stride=patch_size, bias=False)

        if attention:
            # linear layers to compute query/key/value from the patch embedding
            self.q_proj = nn.Linear(patch_size*patch_size, patch_size*patch_size)
            self.k_proj = nn.Linear(patch_size*patch_size, patch_size*patch_size)
            self.v_proj = nn.Linear(patch_size*patch_size, patch_size*patch_size)

    def _patch_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract non‑overlapping patches and flatten them."""
        patches = self.conv(x)  # shape (B,1,H',W')
        return patches.view(x.shape[0], -1)  # (B, num_patches)

    def _attention_weight(self, patches: torch.Tensor) -> torch.Tensor:
        """Compute a self‑attention weighting over the patch embeddings."""
        q = self.q_proj(patches)  # (B, N)
        k = self.k_proj(patches)  # (B, N)
        v = self.v_proj(patches)  # (B, N)
        scores = F.softmax(q @ k.t() / np.sqrt(patches.shape[1]), dim=-1)
        return (scores @ v)  # (B, N)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the hybrid similarity between two batches."""
        x_emb = self._patch_embedding(x)
        y_emb = self._patch_embedding(y)

        if self.attention:
            x_emb = self._attention_weight(x_emb)
            y_emb = self._attention_weight(y_emb)

        # compute pairwise RBF similarity
        diff = x_emb.unsqueeze(1) - y_emb.unsqueeze(0)  # (Bx,By,N)
        sq = (diff**2).sum(dim=-1)  # (Bx,By)
        return torch.exp(-self.gamma * sq)

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                      gamma: float = 1.0, patch_size: int = 2,
                      attention: bool = False) -> np.ndarray:
        """Convenience wrapper that returns a NumPy Gram matrix."""
        model = HybridKernelMethod(gamma, patch_size, attention)
        torch.set_grad_enabled(False)
        gram = torch.zeros(len(a), len(b))
        for i, xi in enumerate(a):
            xi = xi.unsqueeze(0)
            for j, yj in enumerate(b):
                yj = yj.unsqueeze(0)
                gram[i, j] = model(xi, yj).item()
        return gram.numpy()

__all__ = ["HybridKernelMethod"]
