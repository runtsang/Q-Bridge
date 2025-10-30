"""Hybrid self‑attention: classical implementation with optional quantum‑inspired weighting.

The module defines a SelfAttentionHybrid class that implements a multi‑head
self‑attention block.  The attention scores are computed from linear
projections of the input tokens.  When `n_qubits > 0` a learnable
entanglement matrix is added to the scaled dot‑product scores,
mimicking the effect of a small quantum circuit.  The class is
fully PyTorch‑based and can be used inside a transformer
architecture or as a stand‑alone block.

The public factory function `SelfAttention()` returns an instance with
default hyper‑parameters that matches the original seed (embed_dim=4).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionHybrid(nn.Module):
    """Hybrid self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the token embeddings.
    n_qubits : int, default 0
        If >0, a learnable entanglement matrix of shape
        (embed_dim, embed_dim) is added to the attention scores.
    dropout : float, default 0.1
        Dropout probability applied to the attention output.
    """

    def __init__(self, embed_dim: int, n_qubits: int = 0, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)

        # Linear projections for query/key/value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Optional quantum‑inspired entanglement matrix
        if self.n_qubits > 0:
            # Entanglement matrix is a learnable parameter
            self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        else:
            self.entangle = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        batch, seq, _ = x.size()

        # Compute query, key, value
        q = self.q_proj(x)  # (batch, seq, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        # Add quantum‑inspired entanglement if requested
        if self.entangle is not None:
            # Broadcast entangle matrix across batch and seq
            scores = scores + self.entangle.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)
        return attn_out

def SelfAttention() -> SelfAttentionHybrid:
    """Factory returning a default instance matching the original seed."""
    return SelfAttentionHybrid(embed_dim=4, n_qubits=0)

__all__ = ["SelfAttentionHybrid", "SelfAttention"]
