import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionGen064:
    """
    Multi‑head self‑attention with explicit weight matrices supplied via
    ``rotation_params``.  The API matches the quantum counterpart: the
    ``run`` method accepts ``rotation_params``, ``entangle_params`` (ignored
    by the classical variant), and a NumPy input array.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear layers with biases (optional)
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.ln = nn.LayerNorm(embed_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, embed_dim = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch, seq_len, heads * head_dim)

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
            Flat array of length ``3 * embed_dim * embed_dim``.  It contains
            the weight matrices for the query, key and value projections in
            that order.
        entangle_params : np.ndarray
            Ignored by the classical implementation but kept for API
            compatibility.
        inputs : np.ndarray
            Input tensor of shape ``(batch, seq_len, embed_dim)``.
        """
        # Convert to torch tensor
        x = torch.as_tensor(inputs, dtype=torch.float32)

        # Unpack rotation_params into weight matrices
        q_shape = self.embed_dim * self.embed_dim
        q_params = rotation_params[:q_shape].reshape(self.embed_dim, self.embed_dim)
        k_params = rotation_params[q_shape:2 * q_shape].reshape(self.embed_dim, self.embed_dim)
        v_params = rotation_params[2 * q_shape:].reshape(self.embed_dim, self.embed_dim)

        # Load the matrices into the linear layers (biases are left untouched)
        with torch.no_grad():
            self.w_q.weight.copy_(torch.as_tensor(q_params))
            self.w_k.weight.copy_(torch.as_tensor(k_params))
            self.w_v.weight.copy_(torch.as_tensor(v_params))

        # Linear projections
        queries = self.w_q(x)
        keys    = self.w_k(x)
        values  = self.w_v(x)

        # Split heads
        queries = self._split_heads(queries)
        keys    = self._split_heads(keys)
        values  = self._split_heads(values)

        # Scaled dot‑product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / np.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        context = torch.matmul(scores, values)
        context = self._combine_heads(context)

        # Final projection
        out = self.w_o(context)

        # Residual connection and layer norm
        out = self.ln(out + x)

        return out.detach().numpy()

__all__ = ["SelfAttentionGen064"]
