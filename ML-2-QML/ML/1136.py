"""Enhanced classical self‑attention with multi‑head support and dropout.

The class mimics the original interface but adds richer functionality:
* multi‑head scaled dot‑product attention
* optional dropout on the attention matrix
* bias terms for Q and K
"""

import numpy as np
import torch
import torch.nn.functional as F

class SelfAttentionModule:
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0, bias: bool = True):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the input embeddings.
        num_heads : int, default 1
            Number of attention heads.
        dropout : float, default 0.0
            Dropout probability applied to the attention scores.
        bias : bool, default True
            Whether to add a learnable bias to Q and K projections.
        """
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.bias = bias

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Weight matrices for Q, K and V of shape
            (num_heads, 3, embed_dim, embed_dim).  The second dimension
            selects the linear layer (Q, K, V).
        entangle_params : np.ndarray
            Bias vector for each head of shape (num_heads, head_dim).
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output of the self‑attention layer, same shape as *inputs*.
        """
        batch, seq_len, _ = inputs.shape
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32)

        # Linear projections
        Q = torch.einsum(
            'bse,nhd->bshd',
            inputs_t,
            torch.as_tensor(rotation_params[:, 0], dtype=torch.float32),
        )
        K = torch.einsum(
            'bse,nhd->bshd',
            inputs_t,
            torch.as_tensor(rotation_params[:, 1], dtype=torch.float32),
        )
        V = torch.einsum(
            'bse,nhd->bshd',
            inputs_t,
            torch.as_tensor(rotation_params[:, 2], dtype=torch.float32),
        )

        if self.bias:
            bias = torch.as_tensor(entangle_params, dtype=torch.float32)
            Q += bias.unsqueeze(0).unsqueeze(1)
            K += bias.unsqueeze(0).unsqueeze(1)

        # Scaled dot‑product
        scores = torch.einsum('bshd,bshd->bhnm', Q, K) / np.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)
        if self.dropout > 0.0:
            scores = F.dropout(scores, p=self.dropout, training=True)

        # Weighted sum
        out = torch.einsum('bhnm,bshd->bshd', scores, V)
        out = out.reshape(batch, seq_len, self.embed_dim)
        return out.numpy()

def SelfAttention():
    """
    Factory returning a SelfAttentionModule configured with 4‑dimensional
    embeddings, two heads and 10 % dropout.  The signature matches the
    original ``SelfAttention`` function so that existing code continues to
    import ``SelfAttention`` from this module.
    """
    return SelfAttentionModule(embed_dim=4, num_heads=2, dropout=0.1)

__all__ = ["SelfAttention"]
