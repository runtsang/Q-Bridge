"""Extended multi‑head self‑attention with dropout and residual connections.

The class mirrors the interface of the original seed but adds
multi‑head capability, layer‑norm, and dropout, making it
directly usable in transformer stacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionBlock(nn.Module):
    """Multi‑head self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability applied to the attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor | None = None,
        entangle_params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the self‑attention output.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, seq_len, embed_dim).
        rotation_params : torch.Tensor, optional
            Dummy argument kept for API compatibility with the quantum version.
        entangle_params : torch.Tensor, optional
            Dummy argument kept for API compatibility with the quantum version.

        Returns
        -------
        torch.Tensor
            Shape (batch, seq_len, embed_dim). The output of the attention block.
        """
        # Linear projections
        qkv = self.qkv(inputs)  # (batch, seq_len, 3*embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi‑head
        def reshape(x):
            return x.view(inputs.shape[0], inputs.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        # Scaled dot‑product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(inputs.shape[0], inputs.shape[1], self.embed_dim)

        # Output projection + residual + norm
        out = self.out_proj(attn_output)
        out = self.norm(out + inputs)
        return out

__all__ = ["SelfAttentionBlock"]
