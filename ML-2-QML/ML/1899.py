import numpy as np
import torch
import torch.nn.functional as F

class SelfAttention:
    """
    Multi‑head self‑attention with optional dropout.
    Mirrors the original interface but now supports
    embedding dimensions divisible by `num_heads` and
    stochastic regularisation.
    """
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

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
            Weight matrix for the query linear map.
            Shape: (num_heads, head_dim, input_dim)
        entangle_params : np.ndarray
            Weight matrix for the key linear map.
            Shape: (num_heads, head_dim, input_dim)
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, input_dim).

        Returns
        -------
        np.ndarray
            Attention output of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, input_dim = inputs.shape
        if rotation_params.shape!= (self.num_heads, self.head_dim, input_dim):
            raise ValueError("rotation_params shape mismatch")
        if entangle_params.shape!= (self.num_heads, self.head_dim, input_dim):
            raise ValueError("entangle_params shape mismatch")

        x = torch.tensor(inputs, dtype=torch.float32)
        q = torch.einsum('bsi,hdi->bhds', x, rotation_params)
        k = torch.einsum('bsi,hdi->bhds', x, entangle_params)
        v = x.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        scores = torch.einsum('bhds,bhdt->bhst', q, k) / np.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)

        if self.dropout > 0.0:
            scores = F.dropout(scores, p=self.dropout, training=True)

        out = torch.einsum('bhst,bhdt->bhds', scores, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, self.embed_dim)
        return out.numpy()

__all__ = ["SelfAttention"]
