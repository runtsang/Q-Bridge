"""Advanced multi‑head self‑attention implemented with PyTorch.

The class mirrors the original interface but adds:
* configurable number of heads
* optional dropout before attention scores
* support for batched inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AdvancedSelfAttention(nn.Module):
    """
    Multi‑head self‑attention with dropout.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.0
        Dropout probability applied to the attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute self‑attention.

        Parameters
        ----------
        inputs : Tensor, shape (batch, seq, embed_dim)
            Input embeddings.
        rotation_params : Tensor
            Unused in this implementation but kept for API compatibility.
        entangle_params : Tensor
            Unused in this implementation but kept for API compatibility.

        Returns
        -------
        Tensor
            The attended representation.
        """
        batch, seq, _ = inputs.size()

        # Project to Q, K, V
        Q = self.q_proj(inputs)
        K = self.k_proj(inputs)
        V = self.v_proj(inputs)

        # Reshape to (batch, heads, seq, head_dim)
        Q = Q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, V)

        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)

        # Final linear projection
        return self.out_proj(attended)


__all__ = ["AdvancedSelfAttention"]
