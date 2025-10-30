"""Enhanced classical multi‑head self‑attention module.

Implements a PyTorch :class:`torch.nn.Module` that mirrors the
quantum interface while providing trainable multi‑head attention.
The ``run`` method keeps the legacy signature
``run(rotation_params, entangle_params, inputs)`` so existing
scripts continue to work, but internally the weights are
learned during training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionEnhanced(nn.Module):
    """
    Multi‑head self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability applied to the attention weights.
    """

    def __init__(self, embed_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.scale = self.head_dim ** -0.5

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of same shape as ``x``.
        """
        B, N, _ = x.shape

        # Linear projections
        q = self.Wq(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, self.embed_dim)
        out = self.out_proj(context)
        return out

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Compatibility wrapper that accepts legacy *rotation_params* and
        *entangle_params* tensors. These are interpreted as weight
        matrices for the query and key projections respectively.
        The value projection stays identity.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape ``(embed_dim, embed_dim)`` – used to overwrite ``Wq``.
        entangle_params : np.ndarray
            Shape ``(embed_dim, embed_dim)`` – used to overwrite ``Wk``.
        inputs : np.ndarray
            Input data of shape ``(batch, seq_len, embed_dim)``.

        Returns
        -------
        np.ndarray
            The attention output as a NumPy array.
        """
        # Overwrite the linear layers with provided parameters
        with torch.no_grad():
            self.Wq.weight.copy_(torch.tensor(rotation_params, dtype=torch.float32))
            self.Wk.weight.copy_(torch.tensor(entangle_params, dtype=torch.float32))

        x = torch.as_tensor(inputs, dtype=torch.float32)
        out = self.forward(x)
        return out.detach().cpu().numpy()

__all__ = ["SelfAttentionEnhanced"]
