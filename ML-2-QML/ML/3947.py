"""Hybrid classical self‑attention with optional quantum‑inspired embedding.

The class extends the basic self‑attention mechanism by incorporating a lightweight
quantum‑inspired embedding that mimics the behaviour of a parametric quantum circuit.
This allows the model to be used in a purely classical training pipeline while still
benefiting from the expressive power of quantum feature maps.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSelfAttentionML(nn.Module):
    """
    Classical self‑attention module with optional quantum‑inspired embedding.
    """
    def __init__(self, embed_dim: int, use_qembedding: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_qembedding = use_qembedding
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        if use_qembedding:
            # Simple quantum‑inspired mapping: sin/cos of input reshaped to match embed_dim
            self.qembed = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Self‑attention output of shape (batch, seq_len, embed_dim).
        """
        if self.use_qembedding:
            x = self.qembed(x)
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, values)

def SelfAttention():
    """
    Public factory that mirrors the original interface.
    Returns an instance of :class:`HybridSelfAttentionML`.
    """
    return HybridSelfAttentionML(embed_dim=4, use_qembedding=True)
