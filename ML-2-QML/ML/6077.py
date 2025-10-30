import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridSamplerAttention(nn.Module):
    """
    Classical hybrid sampler‑attention module.
    First samples a 2‑dimensional distribution via a small neural network,
    then refines the representation with a self‑attention block.
    """

    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.sampler = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 2)
        )
        self.embed_dim = embed_dim
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, embed_dim).
        """
        probs = F.softmax(self.sampler(inputs), dim=-1)
        query = probs @ self.rotation
        key = probs @ self.entangle
        scores = F.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        output = scores @ probs
        return output

__all__ = ["HybridSamplerAttention"]
