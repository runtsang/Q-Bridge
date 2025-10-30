import torch
import torch.nn.functional as F
import numpy as np

class SelfAttentionModel:
    """
    Multi‑head self‑attention module with configurable dropout.
    Parameters
    ----------
    embed_dim : int
        Dimensionality of input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability applied to the attention scores.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        # Weight matrices for each head
        self.W_q = torch.nn.Parameter(torch.randn(num_heads, embed_dim, self.head_dim))
        self.W_k = torch.nn.Parameter(torch.randn(num_heads, embed_dim, self.head_dim))
        self.W_v = torch.nn.Parameter(torch.randn(num_heads, embed_dim, self.head_dim))
        self.W_o = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))

        self.drop = torch.nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim) and transpose
        for efficient batch computation.
        """
        batch, seq_len, embed = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (batch, heads, seq_len, head_dim)

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the multi‑head attention output.
        Parameters
        ----------
        inputs : np.ndarray of shape (batch, seq_len, embed_dim)
            Input embeddings.
        rotation_params : np.ndarray
            Shape (num_heads, embed_dim, head_dim) – used as query weight.
        entangle_params : np.ndarray
            Shape (num_heads, embed_dim, head_dim) – used as key weight.
        Returns
        -------
        np.ndarray
            Attention‑weighted representation of the input.
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)
        # Apply head‑specific linear projections
        Q = torch.einsum("bse,heh->bheh", x, rotation_params)   # (batch, heads, seq_len, head_dim)
        K = torch.einsum("bse,heh->bheh", x, entangle_params)
        V = self._split_heads(x)                                 # (batch, heads, seq_len, head_dim)

        # Compute scaled dot‑product attention per head
        scores = torch.einsum("bheh,bkeh->bhek", Q, K) / np.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)
        scores = self.drop(scores)

        # Weighted sum over values
        head_out = torch.einsum("bhek,bkeh->bheh", scores, V)

        # Concatenate heads and project back to embed_dim
        head_out = head_out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.embed_dim)
        output = torch.einsum("bse,eh->bsh", head_out, self.W_o)

        return output.detach().numpy()

__all__ = ["SelfAttentionModel"]
