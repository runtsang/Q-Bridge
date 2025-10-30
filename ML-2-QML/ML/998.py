import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention module that extends the original seed by adding
    dropout, residual connections, and optional external weight matrices.
    The public interface remains compatible: ``run(rotation_params,
    entangle_params, inputs)``.
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

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor,
                rotation_params: np.ndarray | None = None,
                entangle_params: np.ndarray | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, seq_len, embed_dim).
        rotation_params : np.ndarray, optional
            Weight matrix for Q projection. Shape (embed_dim, embed_dim).
        entangle_params : np.ndarray, optional
            Weight matrix for K projection. Shape (embed_dim, embed_dim).

        Returns
        -------
        torch.Tensor
            Shape (batch, seq_len, embed_dim) – the attended representation.
        """
        if rotation_params is not None:
            self.q_proj.weight.data = torch.from_numpy(rotation_params).float()
        if entangle_params is not None:
            self.k_proj.weight.data = torch.from_numpy(entangle_params).float()

        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)

        # Reshape for multi‑head attention
        batch, seq_len, _ = q.size()
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        out = self.out_proj(out)
        return out

__all__ = ["SelfAttention"]
