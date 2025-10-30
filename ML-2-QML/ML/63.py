"""Enhanced multi‑head self‑attention implemented with PyTorch.

The class mirrors the quantum interface but adds:
* configurable number of heads
* dropout after attention weighting
* positional encoding added to the query/key/value tensors
"""

import numpy as np
import torch
import torch.nn.functional as F


class SelfAttention:
    """Multi‑head self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the token embeddings.
    num_heads : int, default 1
        Number of attention heads.
    dropout : float, default 0.0
        Dropout probability applied to the attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = torch.nn.Dropout(dropout)

        # Linear projections for queries, keys, values
        self.W_q = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = torch.nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.W_o = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    def _positional_encoding(self, seq_len: int) -> torch.Tensor:
        """Simple sinusoidal positional encoding."""
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        dim = torch.arange(self.embed_dim, dtype=torch.float32).unsqueeze(0)
        div_term = torch.exp(dim * -(np.log(10000.0) / self.embed_dim))
        pe = torch.zeros(seq_len, self.embed_dim)
        pe[:, 0::2] = torch.sin(pos * div_term[0::2])
        pe[:, 1::2] = torch.cos(pos * div_term[1::2])
        return pe

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the multi‑head self‑attention output.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (num_heads, embed_dim, embed_dim). Used as the weight matrices for Q/K/V.
        entangle_params : np.ndarray
            Shape (num_heads, embed_dim, embed_dim). Not used in the classical version but kept for API compatibility.
        inputs : np.ndarray
            Shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = inputs.shape
        device = torch.device("cpu")

        # Convert to torch tensors
        x = torch.as_tensor(inputs, dtype=torch.float32, device=device)

        # Add positional encoding
        pe = self._positional_encoding(seq_len).to(device)
        x = x + pe

        # Apply rotation params as linear projections
        # For each head we create a projection matrix from rotation_params
        q = torch.zeros(batch, seq_len, self.embed_dim, device=device)
        k = torch.zeros(batch, seq_len, self.embed_dim, device=device)
        v = torch.zeros(batch, seq_len, self.embed_dim, device=device)

        for h in range(self.num_heads):
            Wq_h = torch.as_tensor(rotation_params[h], dtype=torch.float32, device=device)
            Wk_h = torch.as_tensor(entangle_params[h], dtype=torch.float32, device=device)
            Wv_h = torch.as_tensor(rotation_params[h], dtype=torch.float32, device=device)

            q_h = torch.matmul(x, Wq_h.t())
            k_h = torch.matmul(x, Wk_h.t())
            v_h = torch.matmul(x, Wv_h.t())

            q[:, :, h * self.head_dim : (h + 1) * self.head_dim] = q_h
            k[:, :, h * self.head_dim : (h + 1) * self.head_dim] = k_h
            v[:, :, h * self.head_dim : (h + 1) * self.head_dim] = v_h

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        # Final linear projection
        out = self.W_o(out)
        return out.detach().cpu().numpy()
__all__ = ["SelfAttention"]
