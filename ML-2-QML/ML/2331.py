"""Hybrid classical self‑attention with trainable feed‑forward regressor.

The class implements a standard scaled‑dot‑product attention mechanism
followed by a lightweight neural network.  It is fully differentiable
and can be integrated into larger PyTorch models.
"""

import torch
import torch.nn as nn
import numpy as np

class HybridSelfAttention(nn.Module):
    """
    Classical hybrid self‑attention model.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    hidden_dim : int
        Size of the hidden layer in the regressor.
    """

    def __init__(self, embed_dim: int = 4, hidden_dim: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        # Simple regressor mirroring EstimatorQNN
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute self‑attention followed by regression.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch, 1).
        """
        q = self.query(inputs)          # (B, L, E)
        k = self.key(inputs)            # (B, L, E)
        v = self.value(inputs)          # (B, L, E)

        scores = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1
        )  # (B, L, L)

        attn_out = torch.matmul(scores, v)  # (B, L, E)
        # aggregate over sequence dimension (mean)
        attn_out = attn_out.mean(dim=1)     # (B, E)

        return self.regressor(attn_out)

def get_hybrid_self_attention(embed_dim: int = 4, hidden_dim: int = 8):
    """
    Factory function returning a ready‑to‑use HybridSelfAttention instance.
    """
    return HybridSelfAttention(embed_dim=embed_dim, hidden_dim=hidden_dim)
