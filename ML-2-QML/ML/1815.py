"""Enhanced multi‑head self‑attention module.

The class exposes the same API as the seed but adds support for multiple heads, dropout and an optional residual connection. Rotation and entangle parameters are treated as weight matrices for the query, key and value projections. The implementation is fully PyTorch‑based and can be dropped into any pipeline that expects a callable ``run`` method.

"""

import torch
import torch.nn.functional as F
import numpy as np

class SelfAttention:
    """
    Multi‑head self‑attention compatible with the original seed interface.

    Parameters
    ----------
    embed_dim : int
        Dimension of the input embeddings.
    num_heads : int, optional
        Number of attention heads. Must divide ``embed_dim``.
    dropout : float, optional
        Dropout probability applied to the attention weights.
    residual : bool, optional
        If True, add a residual connection from the inputs to the output.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 1,
                 dropout: float = 0.1,
                 residual: bool = True):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.dropout = dropout
        self.residual = residual

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Execute a forward pass of the multi‑head self‑attention.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of shape ``(embed_dim, embed_dim)`` containing the linear
            weight matrix for the query projection.
        entangle_params : np.ndarray
            Array of shape ``(embed_dim, embed_dim)`` containing the linear
            weight matrix for the key projection.
        inputs : np.ndarray
            Input tensor of shape ``(batch, seq_len, embed_dim)`` or
            ``(seq_len, embed_dim)``. The implementation accepts the latter
            and broadcasts to a batch of size 1.

        Returns
        -------
        outputs : np.ndarray
            Tensor of the same shape as ``inputs`` containing the
            attention‑weighted representations.
        """
        if inputs.ndim == 2:  # (seq_len, embed_dim)
            inputs = inputs[None, :, :]  # add batch dimension

        batch, seq_len, _ = inputs.shape

        # Linear projections
        query = torch.mm(inputs, torch.as_tensor(rotation_params, dtype=torch.float32))
        key   = torch.mm(inputs, torch.as_tensor(entangle_params, dtype=torch.float32))
        value = torch.as_tensor(inputs, dtype=torch.float32)

        # Reshape to (batch, seq_len, num_heads, head_dim)
        query = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key   = key.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)
        scores = F.dropout(scores, p=self.dropout, training=False)

        # Weighted sum of values
        context = torch.matmul(scores, value)  # (batch, num_heads, seq_len, head_dim)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        if self.residual:
            context = context + inputs

        return context.numpy().squeeze(0) if inputs.shape[0] == 1 else context.numpy()
