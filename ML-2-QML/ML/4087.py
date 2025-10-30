"""Hybrid classical self‑attention with convolution and RBF kernel."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Tuple


class RBFKernel(nn.Module):
    """Radial‑basis function kernel with learnable gamma."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y : (B, L, D)
        diff = x.unsqueeze(2) - y.unsqueeze(1)          # (B, L, L, D)
        dist2 = torch.sum(diff ** 2, dim=-1)            # (B, L, L)
        return torch.exp(-self.gamma * dist2)           # (B, L, L)


class HybridSelfAttention(nn.Module):
    """
    Classical self‑attention that first applies a 2‑D convolution
    to each embedding, then computes attention weights using a
    RBF kernel on query/key pairs.
    """
    def __init__(self,
                 embed_dim: int = 4,
                 conv_kernel: int = 2,
                 gamma: float = 1.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(1, 1, kernel_size=conv_kernel, bias=True)
        self.kernel = RBFKernel(gamma)

    def forward(self,
                inputs: torch.Tensor,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape (B, L, D) where B=batch, L=sequence length, D=embed_dim.
        rotation_params, entangle_params : np.ndarray
            Parameters for query/key projection.
            Expected shape (3*D,).
        Returns
        -------
        torch.Tensor
            Output of shape (B, L, D).
        """
        B, L, D = inputs.shape
        # Convolution on each embedding vector reshaped to a square
        conv_out = []
        for i in range(B):
            vec = inputs[i].reshape(1, 1, D, D)
            conv_val = self.conv(vec).mean().item()
            conv_out.append(conv_val)
        conv_out = torch.tensor(conv_out, device=inputs.device).unsqueeze(-1)  # (B,1)

        # Query, key, value projections
        rotation = torch.from_numpy(rotation_params.reshape(D, -1)).float()
        entangle = torch.from_numpy(entangle_params.reshape(D, -1)).float()

        query = torch.matmul(inputs, rotation)          # (B, L, K)
        key   = torch.matmul(inputs, entangle)          # (B, L, K)
        value = inputs                                   # (B, L, D)

        # Kernel‑based similarity
        sim = self.kernel(query, key)                    # (B, L, L)
        scores = torch.softmax(sim, dim=-1)              # (B, L, L)

        attn_out = torch.matmul(scores, value)          # (B, L, D)
        # Add convolution scalar to each token
        output = attn_out + conv_out.unsqueeze(1)        # broadcast to (B, L, D)
        return output


__all__ = ["HybridSelfAttention"]
