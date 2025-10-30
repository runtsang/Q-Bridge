"""Hybrid classical sampler with self‑attention mechanism.

The module defines SamplerQNNGen187 which first applies a
classical self‑attention block to the input embeddings and then
feeds the attention scores into a small feed‑forward network that
produces a probability distribution over two outcomes.  The
attention block is inspired by the SelfAttention seed, while the
sampler network is a direct adaptation of SamplerQNN.

The design keeps the model fully classical (PyTorch) and is
compatible with the original SamplerQNN interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Utility self‑attention used by SamplerQNNGen187."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        # Linear layers to generate query/key/value projections
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape (batch, embed_dim)
        Returns:
            Tensor of shape (batch, embed_dim) – attention weighted values.
        """
        query = self.query_proj(inputs)
        key   = self.key_proj(inputs)
        value = self.value_proj(inputs)

        scores = F.softmax(query @ key.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, value)

class SamplerQNNGen187(nn.Module):
    """
    Classical hybrid sampler that first computes self‑attention over the
    input embeddings and then passes the attention output through a
    small feed‑forward network that mimics a quantum sampler.

    The architecture mirrors the original SamplerQNN but replaces the
    first linear layer with an attention block, thereby giving the
    model a richer representation of the input.
    """
    def __init__(self, embed_dim: int = 4, hidden_dim: int = 4):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim)
        # Sampler network (adapted from SamplerQNN)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape (batch, embed_dim)
        Returns:
            Probability distribution over two outcomes.
        """
        attn_out = self.attention(inputs)
        logits   = self.net(attn_out)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNNGen187"]
