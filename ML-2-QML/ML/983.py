"""Enhanced classical self‑attention with multi‑head, dropout, and activation support."""
import numpy as np
import torch
import torch.nn.functional as F

class SelfAttention:
    """
    Classical self‑attention supporting multiple heads, dropout, and a choice of activation.
    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default=1
        Number of attention heads.
    dropout : float, default=0.0
        Dropout probability applied to the attention weights.
    activation : str or callable, default="relu"
        Activation function applied to the value projections. Supported strings:'relu', 'gelu'.
    """
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0,
                 activation: str | callable = "relu"):
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.activation = activation if isinstance(activation, str) else activation

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.activation, str):
            if self.activation.lower() == "relu":
                return F.relu(x)
            elif self.activation.lower() == "gelu":
                return F.gelu(x)
            else:
                raise ValueError(f"Unsupported activation: {self.activation}")
        return self.activation(x)

    def run(self,
            inputs: np.ndarray,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray) -> np.ndarray:
        """
        Compute the self‑attention output.
        Parameters
        ----------
        inputs : np.ndarray
            Input sequence of shape (seq_len, embed_dim).
        rotation_params : np.ndarray
            Parameters used to compute query projections. Shape (embed_dim, embed_dim).
        entangle_params : np.ndarray
            Parameters used to compute key projections. Shape (embed_dim, embed_dim).
        Returns
        -------
        np.ndarray
            Output sequence of shape (seq_len, embed_dim).
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)
        # Linear projections via provided parameter matrices
        q = x @ rotation_params.reshape(self.embed_dim, -1)
        k = x @ entangle_params.reshape(self.embed_dim, -1)
        v = x
        # Reshape for multi‑head
        seq_len = x.shape[0]
        q = q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        # Scaled dot‑product attention
        scores = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim), dim=-1)
        if self.dropout > 0.0:
            scores = torch.nn.functional.dropout(scores, p=self.dropout, training=True)
        out = scores @ v
        out = out.transpose(0, 1).contiguous().view(seq_len, self.embed_dim)
        out = self._apply_activation(out)
        return out.numpy()

__all__ = ["SelfAttention"]
