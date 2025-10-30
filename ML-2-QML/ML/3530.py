"""Hybrid classical sampler that incorporates self‑attention."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalSelfAttention:
    """Light‑weight self‑attention module compatible with NumPy input."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        q = torch.tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        k = torch.tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        v = torch.tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(q @ k.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ v).numpy()


class SamplerAttentionQNN(nn.Module):
    """Classical sampler that uses self‑attention as a feature extractor."""
    def __init__(self, embed_dim: int = 4, hidden_dim: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = ClassicalSelfAttention(embed_dim)

        # Learnable attention parameters
        self.rots = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.ent = nn.Parameter(torch.randn(embed_dim, embed_dim))

        # Sampler head
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Convert to NumPy for the attention helper
        rot_np = self.rots.detach().cpu().numpy()
        ent_np = self.ent.detach().cpu().numpy()
        attn_out = self.attention.run(rot_np, ent_np, inputs.cpu().numpy())

        # Pass attention output through the sampler head
        attn_tensor = torch.tensor(attn_out, dtype=torch.float32, device=inputs.device)
        logits = self.net(attn_tensor)
        return F.softmax(logits, dim=-1)


__all__ = ["SamplerAttentionQNN"]
