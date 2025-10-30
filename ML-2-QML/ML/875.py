import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionModule(nn.Module):
    """Multi‑head self‑attention with optional dropout and residual connection.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.0
        Dropout probability applied to attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections for query, key, value
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, rotation_params: torch.Tensor, entangle_params: torch.Tensor) -> torch.Tensor:
        """
        Compute multi‑head self‑attention. The rotation_params and entangle_params are
        treated as learned weight matrices that will be fused into the qkv projection
        during training.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        rotation_params : torch.Tensor
            Tensor of shape (embed_dim, embed_dim) used to initialize the query matrix.
        entangle_params : torch.Tensor
            Tensor of shape (embed_dim, embed_dim) used to initialize the key matrix.

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as ``x``.
        """
        batch, seq_len, _ = x.size()

        # Initialize query and key matrices from params
        q_proj = rotation_params.reshape(self.embed_dim, self.embed_dim)
        k_proj = entangle_params.reshape(self.embed_dim, self.embed_dim)

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Apply custom projections
        q = torch.matmul(q, q_proj)
        k = torch.matmul(k, k_proj)

        # Reshape for multi‑head
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (batch, heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        # Residual + output projection
        out = self.out_proj(out)
        return out + x

__all__ = ["SelfAttentionModule"]
