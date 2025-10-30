import numpy as np
import torch
from typing import Optional

class SelfAttention:
    """Multi‑head self‑attention with learnable linear projections.

    The implementation extends the original single‑head design by
    supporting an arbitrary number of heads.  The linear layers are
    populated from the ``rotation_params`` and ``entangle_params``
    arguments, which are supplied as NumPy arrays matching the
    weight shape of the projection matrices.  The value projection
    is fixed to the identity to keep the interface compact.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, head_dim: Optional[int] = None):
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or embed_dim // num_heads

        self.W_q = torch.nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.W_k = torch.nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.W_v = torch.nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Execute a forward pass of the attention block.

        Parameters
        ----------
        rotation_params : np.ndarray
            Weight matrix for the query projection.
            Shape: (embed_dim, num_heads * head_dim)
        entangle_params : np.ndarray
            Weight matrix for the key projection.
            Shape: (embed_dim, num_heads * head_dim)
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output representations of shape (batch, seq_len, embed_dim).
        """
        # Load the supplied parameters into the linear layers
        self.W_q.weight.data = torch.as_tensor(rotation_params, dtype=torch.float32)
        self.W_k.weight.data = torch.as_tensor(entangle_params, dtype=torch.float32)
        # Value projection uses the identity matrix
        self.W_v.weight.data = torch.eye(self.num_heads * self.head_dim, dtype=torch.float32)

        x = torch.as_tensor(inputs, dtype=torch.float32)

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = self._reshape(q)
        k = self._reshape(k)
        v = self._reshape(v)

        scale = 1.0 / np.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)

        return out.detach().cpu().numpy()
