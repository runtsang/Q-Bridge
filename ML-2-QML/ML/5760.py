"""Hybrid estimator combining self‑attention and feed‑forward regression.

This module defines HybridEstimator, a PyTorch nn.Module that first applies a
classical self‑attention block to the input features and then passes the
attended representation through a small feed‑forward network.  The design
mirrors EstimatorQNN while adding an attention mechanism inspired by
SelfAttention.
"""

import torch
from torch import nn
import numpy as np


class ClassicalSelfAttention:
    """Light‑weight self‑attention implementation."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class HybridEstimator(nn.Module):
    """Classical hybrid estimator."""
    def __init__(self, input_dim: int, embed_dim: int = 4, hidden_dims: list[int] = (8, 4)):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to numpy for attention, then back to tensor
        attn_out = self.attention.run(
            rotation_params=np.random.randn(self.attention.embed_dim * self.attention.embed_dim),
            entangle_params=np.random.randn(self.attention.embed_dim * (self.attention.embed_dim - 1)),
            inputs=x.detach().cpu().numpy()
        )
        attn_tensor = torch.from_numpy(attn_out).to(x.device, dtype=x.dtype)
        return self.feedforward(attn_tensor)


__all__ = ["HybridEstimator"]
