"""Enhanced multi‑head self‑attention with optional dropout and bias.

The module mirrors the original interface but adds support for multiple
attention heads, dropout, and bias terms.  The linear projections are
constructed from the supplied rotation and entangle parameters, enabling
easy swapping with the quantum counterpart.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default=1
        Number of attention heads.
    dropout : float, default=0.0
        Dropout probability applied to the attention weights.
    bias : bool, default=True
        Whether to include bias terms in the linear projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.bias = bias

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [batch, seq_len, embed_dim] → [batch, num_heads, seq_len, head_dim]."""
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the self‑attention output.

        Parameters
        ----------
        rotation_params : np.ndarray
            Flattened weight matrix for Q, K, V. Shape must be
            ``(3 * embed_dim, embed_dim)``.
        entangle_params : np.ndarray
            Biases for the linear layers if ``bias`` is True. Shape
            ``(3 * embed_dim,)``.
        inputs : np.ndarray
            Input batch of shape ``(batch, seq_len, embed_dim)``.

        Returns
        -------
        np.ndarray
            Attention‑weighted representations of shape
            ``(batch, seq_len, embed_dim)``.
        """
        device = torch.device("cpu")
        x = torch.as_tensor(inputs, dtype=torch.float32, device=device)

        # Build linear layers from the supplied parameters
        W = torch.as_tensor(
            rotation_params.reshape(3 * self.embed_dim, self.embed_dim),
            dtype=torch.float32,
            device=device,
        )
        if self.bias:
            b = torch.as_tensor(entangle_params, dtype=torch.float32, device=device)
        else:
            b = torch.zeros(3 * self.embed_dim, dtype=torch.float32, device=device)

        # Linear projections
        qkv = F.linear(x, W, bias=b)  # shape: [batch, seq_len, 3*embed_dim]
        qkv = qkv.reshape(x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 3, 0, 1, 4)  # [3, heads, batch, seq_len, head_dim]
        Q, K, V = qkv[0], qkv[1], qkv[2]  # each: [heads, batch, seq_len, head_dim]

        # Scaled dot‑product attention
        scores = torch.einsum("hbqd,hbkd->hbqk", Q, K) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        out = torch.einsum("hbqk,hbkd->hbqd", attn, V)
        out = out.permute(2, 3, 0, 1).reshape(
            x.shape[0], x.shape[1], self.embed_dim
        )
        return out.detach().cpu().numpy()


def SelfAttention():
    """Convenience factory matching the original seed signature."""
    return SelfAttention(embed_dim=4, num_heads=1, dropout=0.0, bias=True)


__all__ = ["SelfAttention"]
