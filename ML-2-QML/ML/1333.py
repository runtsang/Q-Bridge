import numpy as np
import torch
import torch.nn.functional as F

class SelfAttention:
    """
    Classical multi‑head self‑attention module.
    Parameters
    ----------
    embed_dim : int
        Dimensionality of input embeddings.
    num_heads : int, default 1
        Number of attention heads.
    dropout : float, default 0.0
        Dropout probability applied to attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Linear projections for query, key, value
        self.W_q = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Compute self‑attention over ``inputs``.
        The ``rotation_params`` and ``entangle_params`` arguments are kept for
        interface compatibility with the quantum counterpart but are ignored
        by the classical implementation.

        Parameters
        ----------
        rotation_params : np.ndarray
            Placeholder for quantum rotation parameters.
        entangle_params : np.ndarray
            Placeholder for quantum entanglement parameters.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Attention‑weighted representations of shape (batch, seq_len,
            embed_dim).
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)

        # Project to queries, keys, values
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi‑head attention
        batch, seq_len, _ = Q.shape
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)
        if self.dropout > 0.0:
            scores = F.dropout(scores, p=self.dropout, training=False)

        # Weighted sum
        out = torch.matmul(scores, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        return out.numpy()

__all__ = ["SelfAttention"]
