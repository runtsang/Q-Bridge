"""Enhanced classical self‑attention module with multi‑head support.

The implementation builds on the original seed by adding a true
multi‑head attention mechanism, dropout, and a clean PyTorch
interface that can be dropped into a larger encoder stack.
"""

import torch
import numpy as np

class SelfAttentionModel(torch.nn.Module):
    """
    Multi‑head self‑attention layer.
    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 2
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability applied to the attention weights.
    """
    def __init__(self, embed_dim: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Linear projections for query, key, value
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)

        # Multi‑head attention
        self.attn = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Output projection
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute multi‑head self‑attention.

        Parameters
        ----------
        rotation_params : np.ndarray
            Weight matrix for the query projection (shape: embed_dim × embed_dim).
        entangle_params : np.ndarray
            Weight matrix for the key projection (shape: embed_dim × embed_dim).
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Attention output of shape (batch, seq_len, embed_dim).
        """
        # Convert inputs to torch tensor
        x = torch.as_tensor(inputs, dtype=torch.float32)

        # Project using supplied parameter matrices
        q = torch.matmul(x, torch.as_tensor(rotation_params, dtype=torch.float32))
        k = torch.matmul(x, torch.as_tensor(entangle_params, dtype=torch.float32))
        v = x

        # Multi‑head attention
        attn_output, _ = self.attn(q, k, v)

        # Final linear projection
        out = self.out_proj(attn_output)
        return out.detach().cpu().numpy()

__all__ = ["SelfAttentionModel"]
