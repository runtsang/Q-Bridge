import torch
import torch.nn as nn
import numpy as np

class SelfAttentionHybrid(nn.Module):
    """
    Classical self‑attention module with optional refinement via a feed‑forward network.
    The module replaces the fixed linear projections of the seed with learnable
    weight matrices, allowing gradient‑based optimization.
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        # Linear projections for query, key, value
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # Optional refinement network
        self.refine = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled dot‑product attention and optional refinement.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim) with attended representations.
        """
        Q = self.query_proj(inputs)          # (B, S, D)
        K = self.key_proj(inputs)            # (B, S, D)
        V = self.value_proj(inputs)          # (B, S, D)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_dim, dtype=Q.dtype))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        # Refine the attention output with a small network
        refined = self.refine(attn_output)
        return attn_output + refined

__all__ = ["SelfAttentionHybrid"]
