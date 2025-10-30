import torch
import torch.nn as nn
import numpy as np

class SelfAttentionEnhanced:
    """Classical multi‑head self‑attention with dropout and layer‑normalisation.

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
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def _reshape(self, x: torch.Tensor):
        # (batch, seq, embed) -> (batch, heads, seq, head_dim)
        return x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)

    def run(self, inputs: np.ndarray, rotation_params: np.ndarray, entangle_params: np.ndarray) -> np.ndarray:
        """
        Forward pass of the attention layer.

        Parameters
        ----------
        inputs : np.ndarray
            Input tensor of shape (batch, seq, embed_dim).
        rotation_params : np.ndarray
            Parameters reshaped into a matrix of shape (embed_dim, embed_dim) for linear projections.
        entangle_params : np.ndarray
            Ignored in the classical implementation but kept for API compatibility.

        Returns
        -------
        np.ndarray
            Output tensor of shape (batch, seq, embed_dim).
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)

        # Linear projections
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Reshape for multi‑head
        q = self._reshape(q)
        k = self._reshape(k)
        v = self._reshape(v)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.embed_dim)
        out = self.out_proj(out)
        out = self.layer_norm(out)

        return out.detach().numpy()
__all__ = ["SelfAttentionEnhanced"]
