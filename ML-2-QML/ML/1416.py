"""Classical self‑attention module with multi‑head support.

This implementation extends the original 4‑dimensional attention
to arbitrary embedding sizes and multiple heads, adding dropout
and a convenient ``run`` API that mirrors the quantum interface.
"""

import numpy as np
import torch


class SelfAttentionModel:
    """
    Classical multi‑head self‑attention.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 1
        Number of attention heads.
    dropout : float, default 0.0
        Dropout probability applied to the attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.q_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``inputs``.
        """
        batch, seq_len, _ = inputs.shape

        # Linear projections
        q = self.q_linear(inputs).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_linear(inputs).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_linear(inputs).reshape(batch, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        out = torch.matmul(attn, v)  # (batch, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)
        out = self.out_linear(out)
        return out

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compatibility wrapper that accepts weight matrices in the same
        shape as the original quantum reference.

        Parameters
        ----------
        rotation_params : np.ndarray
            Weight matrix for the query projection of shape (embed_dim, embed_dim).
        entangle_params : np.ndarray
            Weight matrix for the key projection of shape (embed_dim, embed_dim).
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        # Temporarily replace linear weights
        orig_q = self.q_linear.weight.data.clone()
        orig_k = self.k_linear.weight.data.clone()
        self.q_linear.weight.data = torch.as_tensor(rotation_params, dtype=torch.float32)
        self.k_linear.weight.data = torch.as_tensor(entangle_params, dtype=torch.float32)

        out = self.forward(torch.as_tensor(inputs, dtype=torch.float32))

        # Restore original weights
        self.q_linear.weight.data = orig_q
        self.k_linear.weight.data = orig_k

        return out.numpy()


__all__ = ["SelfAttentionModel"]
