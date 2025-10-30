"""Enhanced multi‑head self‑attention implementation."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Multi‑head scaled dot‑product self‑attention with optional dropout.
    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Drop‑out probability applied to attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi‑head self‑attention.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # shapes: (B, N, num_heads, head_dim)

        scores = torch.einsum("bnhd,bmhd->bnhm", q, k) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bnhm,bmhd->bnhd", attn, v)
        out = out.reshape(B, N, self.embed_dim)
        out = self.out_proj(out)
        return out

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper matching the original interface.
        Parameters
        ----------
        inputs : np.ndarray
            Input array of shape (batch, seq_len, embed_dim).
        Returns
        -------
        np.ndarray
            Attention output of the same shape.
        """
        tensor = torch.as_tensor(inputs, dtype=torch.float32)
        return self.forward(tensor).detach().cpu().numpy()
