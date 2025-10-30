import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Multi‑head self‑attention with learnable linear projections.

    This implementation extends the original seed by adding:
    * Multi‑head support (``heads`` parameter).
    * Separate linear layers for query, key, and value projections.
    * Optional dropout and residual connection.
    * A simple API compatible ``run`` method that accepts the same
      ``rotation_params`` and ``entangle_params`` arguments as the seed
      but ignores them internally.
    """

    def __init__(self, embed_dim: int, heads: int = 1, dropout: float = 0.0):
        super().__init__()
        if embed_dim % heads!= 0:
            raise ValueError("embed_dim must be divisible by heads")
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Compute multi‑head self‑attention.

        Parameters
        ----------
        rotation_params : np.ndarray
            Unused in this implementation but kept for API compatibility.
        entangle_params : np.ndarray
            Unused in this implementation but kept for API compatibility.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Attention‑weighted output of shape (batch, heads, seq_len, 1).
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)
        batch, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi‑head
        q = q.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        # Final projection
        out = self.out_proj(out)  # (batch, seq_len, embed_dim)

        # Collapse embed_dim dimension to 1 for compatibility with the seed
        out = out.mean(dim=2, keepdim=True)  # (batch, seq_len, 1)
        # Expand heads dimension
        out = out.unsqueeze(1).expand(-1, self.heads, -1, -1)  # (batch, heads, seq_len, 1)

        return out.detach().cpu().numpy()
