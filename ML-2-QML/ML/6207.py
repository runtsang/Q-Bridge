"""Extended classical self‑attention module with multi‑head, dropout, and optional parameter‑based projections.

The module keeps the original ``run(rotation_params, entangle_params, inputs)`` signature
while adding a full multi‑head implementation and an option to use the supplied
parameter matrices as linear projections for the query and key.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """A flexible self‑attention layer that mirrors the quantum interface.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default=1
        Number of attention heads.
    dropout : float, default=0.0
        Dropout probability applied to the attention weights.
    use_params : bool, default=True
        If ``True``, the ``rotation_params`` and ``entangle_params`` arguments of
        :meth:`run` are interpreted as linear projections for the query and key.
        When ``False`` the layer falls back to the standard multi‑head
        implementation.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0,
                 use_params: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scaling = self.head_dim ** -0.5
        self.use_params = use_params

        # Standard linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Standard multi‑head scaled dot‑product attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, embed_dim)``.
        mask : torch.Tensor, optional
            Attention mask of shape ``(batch, seq_len, seq_len)`` where
            ``0`` indicates positions to be ignored.
        """
        batch, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)
        return self.out_proj(attn_output)

    def get_attention_weights(self, x: torch.Tensor,
                               mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return the attention weights without the final linear projection."""
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        return F.softmax(attn_scores, dim=-1)

    def run(self, rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """Compatibility wrapper that accepts the same signature as the quantum
        implementation.

        When ``use_params`` is ``True`` the supplied matrices are interpreted as
        linear projections for the query and key.  Otherwise the layer behaves
        like the standard multi‑head attention defined in :meth:`forward`.

        Parameters
        ----------
        rotation_params : np.ndarray
            Matrix of shape ``(embed_dim, embed_dim)`` used as the query
            projection when ``use_params=True``.
        entangle_params : np.ndarray
            Matrix of shape ``(embed_dim, embed_dim)`` used as the key
            projection when ``use_params=True``.
        inputs : np.ndarray
            Input array of shape ``(batch, seq_len, embed_dim)``.
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)

        if self.use_params:
            # Interpret the matrices as linear projections
            q = torch.matmul(x, torch.as_tensor(rotation_params, dtype=torch.float32))
            k = torch.matmul(x, torch.as_tensor(entangle_params, dtype=torch.float32))
            v = x
            scores = torch.softmax(q @ k.transpose(-2, -1) * self.scaling, dim=-1)
            return (scores @ v).numpy()
        else:
            return self.forward(x).numpy()

__all__ = ["SelfAttention"]
