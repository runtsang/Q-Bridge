"""Hybrid self‑attention with classical sampler network.

This module implements a classical self‑attention block that incorporates a
small neural sampler (SamplerQNN) to refine the attention distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SamplerQNN(nn.Module):
    """Tiny sampler network inspired by the original SamplerQNN seed."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(x), dim=-1)

class HybridSelfAttentionML(nn.Module):
    """
    Classical self‑attention block that uses a sampler to smooth the
    attention logits.  Designed to mirror the interface of the quantum
    counterpart.
    """
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.sampler = SamplerQNN()

    def forward(self,
                inputs: torch.Tensor,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape (batch, seq_len, embed_dim)
            rotation_params: Rotation parameter vector (unused, kept for API)
            entangle_params: Entanglement parameter vector (unused, kept for API)
        Returns:
            Tensor of shape (batch, seq_len, embed_dim) – attended representation.
        """
        # Compute standard attention logits
        Q = self.query(inputs)  # (B, S, D)
        K = self.key(inputs)    # (B, S, D)
        V = self.value(inputs)  # (B, S, D)

        scores = torch.softmax((Q @ K.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        # Flatten to feed sampler (B*S, 2) – dummy mapping for demo
        bs = scores.shape[0] * scores.shape[1]
        dummy = torch.randn(bs, 2, device=scores.device)
        sampler_out = self.sampler(dummy).reshape(scores.shape[0], scores.shape[1], 2)
        # Combine sampler output with original scores
        refined_scores = scores * sampler_out[..., 0].unsqueeze(-1)
        refined_scores = refined_scores / refined_scores.sum(dim=-1, keepdim=True)
        return refined_scores @ V

__all__ = ["HybridSelfAttentionML"]
