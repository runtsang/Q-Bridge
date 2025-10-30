"""Classical multi‑head self‑attention module with parameter‑controlled weights.

The class mimics the interface of the original seed while adding
multi‑head capability, dropout, and a convenient ``run`` method that
accepts rotation and entangle parameters to overwrite the linear
weights.  The implementation is fully PyTorch‑based and can be
plugged into any existing training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention module.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability applied to the attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape.
        """
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        # Transpose for dot‑product attention
        q = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, seq_len, heads, head_dim)
        attn_output = attn_output.view(batch, seq_len, self.embed_dim)

        return self.out_proj(attn_output)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper that injects the supplied parameters into the
        linear layers and runs a forward pass.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (embed_dim, embed_dim) – used to overwrite the
            query and key projection matrices.
        entangle_params : np.ndarray
            Shape (embed_dim,) – used as a bias for the value
            projection.  Allows a quantum‑style “entanglement” knob.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output of the attention block.
        """
        if rotation_params.shape!= (self.embed_dim, self.embed_dim):
            raise ValueError(f"rotation_params must be of shape ({self.embed_dim}, {self.embed_dim})")
        if entangle_params.shape!= (self.embed_dim,):
            raise ValueError(f"entangle_params must be of shape ({self.embed_dim},)")

        # Overwrite projection weights
        self.q_proj.weight.data = torch.from_numpy(rotation_params).float()
        self.k_proj.weight.data = torch.from_numpy(rotation_params).float()
        # Apply bias to value projection
        self.v_proj.bias = nn.Parameter(torch.from_numpy(entangle_params).float())

        x = torch.from_numpy(inputs).float()
        out = self.forward(x)
        return out.detach().numpy()
