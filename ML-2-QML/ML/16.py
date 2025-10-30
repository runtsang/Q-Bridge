"""Enhanced classical self‑attention with multi‑head support and dropout."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalSelfAttention(nn.Module):
    """
    Multi‑head scaled dot‑product attention.
    Parameters
    ----------
    embed_dim : int
        Dimensionality of input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability applied to attention weights.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute multi‑head attention.
        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (embed_dim, embed_dim) used to initialise q_proj weights.
        entangle_params : np.ndarray
            Shape (embed_dim, embed_dim) used to initialise k_proj weights.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).
        Returns
        -------
        np.ndarray
            Attention output of shape (batch, seq_len, embed_dim).
        """
        # Initialise projections from provided params
        self.q_proj.weight.data = torch.from_numpy(rotation_params.T).float()
        self.k_proj.weight.data = torch.from_numpy(entangle_params.T).float()
        self.v_proj.weight.data = torch.from_numpy(entangle_params.T).float()  # reuse for simplicity

        x = torch.from_numpy(inputs).float()
        batch, seq_len, _ = x.shape

        # Linear projections
        Q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention calculation: (batch, heads, seq_len, head_dim)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Scaled dot‑product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (batch, heads, seq_len, head_dim)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, self.embed_dim)

        output = self.out_proj(context)
        return output.detach().numpy()


def SelfAttention():
    """
    Factory function mirroring the original API.
    Returns an instance of ClassicalSelfAttention with default parameters.
    """
    return ClassicalSelfAttention(embed_dim=4, num_heads=4, dropout=0.1)


__all__ = ["SelfAttention"]
