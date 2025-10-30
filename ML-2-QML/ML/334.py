import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """Classical multi‑head self‑attention module compatible with the original interface.

    Parameters
    ----------
    embed_dim : int
        Dimension of each token embedding.
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

        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_linear = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that mimics the original ``run`` signature.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        rotation_params : torch.Tensor
            Parameters for the linear projections (unused but kept for API compatibility).
        entangle_params : torch.Tensor
            Parameters for the attention score scaling (unused but kept for API compatibility).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = x.size()

        # Linear projections
        q = self.q_linear(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous() \
           .view(batch, seq_len, self.embed_dim)

        # Residual + LayerNorm
        out = self.norm(x + self.out_linear(attn_output))
        return out

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Compatibility wrapper that accepts NumPy arrays and returns NumPy output.
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)
        out = self.forward(x, torch.as_tensor(rotation_params),
                           torch.as_tensor(entangle_params))
        return out.detach().numpy()

def SelfAttention():
    """Factory returning a ready‑to‑use ``MultiHeadSelfAttention`` instance."""
    return MultiHeadSelfAttention(embed_dim=4, num_heads=2, dropout=0.1)

__all__ = ["SelfAttention"]
