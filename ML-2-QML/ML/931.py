"""Enhanced multi‑head self‑attention module with dropout and residual connections.

This implementation builds on the original SelfAttention helper by adding
multi‑head support, layer‑norm, and dropout.  The public API remains
compatible with the seed (`run(rotation_params, entangle_params, inputs)`),
but the internal computation now performs a full Transformer‑style
attention block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention with optional dropout and residual connection.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability applied to attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the self‑attention output.

        Parameters
        ----------
        rotation_params : np.ndarray
            Weight matrix for the query projection.
            Shape must be (embed_dim, embed_dim).
        entangle_params : np.ndarray
            Weight matrix for the key projection.
            Shape must be (embed_dim, embed_dim).
        inputs : np.ndarray
            Input embeddings of shape (batch, seq_len, embed_dim).
        mask : np.ndarray | None
            Optional causal or padding mask of shape (batch, seq_len, seq_len).

        Returns
        -------
        np.ndarray
            The attended representations of shape (batch, seq_len, embed_dim).
        """
        # Convert inputs to torch tensor
        x = torch.as_tensor(inputs, dtype=torch.float32)

        # Apply linear projections using provided matrices
        # We bypass the module's learned weights to honor the seed API.
        q = torch.matmul(x, torch.as_tensor(rotation_params, dtype=torch.float32))
        k = torch.matmul(x, torch.as_tensor(entangle_params, dtype=torch.float32))
        v = x  # value is the raw input

        # Reshape for multi‑head
        batch, seq_len, _ = q.shape
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        # Residual + LayerNorm
        output = self.norm(context + x)
        return output.detach().cpu().numpy()

    # Alias for compatibility with the seed's run method
    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.forward(rotation_params, entangle_params, inputs, mask)
