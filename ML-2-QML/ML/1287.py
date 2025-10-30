import numpy as np
import torch
import math

class SelfAttention:
    """
    Multi‑head self‑attention with optional residual and dropout.
    The interface matches the original seed: run(rotation_params, entangle_params, inputs).
    rotation_params and entangle_params are treated as linear weight matrices that modulate
    the query/key/value projections, allowing a seamless comparison with the quantum version.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1, residual: bool = True):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.residual = residual

        # Base linear layers – weights will be overridden by rotation_params
        self.W_q = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout_layer = torch.nn.Dropout(dropout)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Weight matrix for query and key projections. Shape (embed_dim, embed_dim).
        entangle_params : np.ndarray
            Weight matrix for value projection. Shape (embed_dim, embed_dim).
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output of the self‑attention layer, same shape as `inputs`.
        """
        # Override linear weights with provided parameters
        self.W_q.weight.data = torch.from_numpy(rotation_params).float()
        self.W_k.weight.data = torch.from_numpy(rotation_params).float()
        self.W_v.weight.data = torch.from_numpy(entangle_params).float()

        x = torch.from_numpy(inputs).float()

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # reshape for multi‑head
        batch, seq, _ = Q.shape
        Q = Q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, L, D)
        K = K.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout_layer(scores)

        context = torch.matmul(scores, V)  # (B, heads, L, D)
        context = context.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        out = self.out_proj(context)
        if self.residual:
            out = out + x
        return out.detach().numpy()

__all__ = ["SelfAttention"]
