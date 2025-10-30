import numpy as np
import torch
from torch import nn

class SelfAttentionHybridML(nn.Module):
    """Hybrid classical attention module mirroring quantum parameterization."""
    def __init__(self, embed_dim: int, attn_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        # Trainable rotation and entanglement matrices
        self.rotation = nn.Parameter(torch.randn(embed_dim, attn_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, attn_dim))
        # Regression head on attention output
        self.estimator = nn.Sequential(
            nn.Linear(attn_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = inputs @ self.rotation
        key   = inputs @ self.entangle
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        attn_output = scores @ inputs
        return self.estimator(attn_output)

__all__ = ["SelfAttentionHybridML"]
