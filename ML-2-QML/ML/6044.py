import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SelfAttention(nn.Module):
    """
    Classical self‑attention module with optional dropout, bias and layer‑norm.
    The API mirrors the original seed but adds a richer set of hyper‑parameters
    and a ``forward`` method compatible with PyTorch training loops.
    """

    def __init__(
        self,
        embed_dim: int = 4,
        dropout: float = 0.1,
        bias: bool = True,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim) if layer_norm else None

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute multi‑head self‑attention over a batch of sequences.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        Q = self.q_proj(inputs)
        K = self.k_proj(inputs)
        V = self.v_proj(inputs)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return out

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compatibility wrapper that ignores the quantum parameters and
        forwards the input to the classical attention block.
        """
        inputs_t = torch.tensor(inputs, dtype=torch.float32)
        return self.forward(inputs_t).detach().numpy()

__all__ = ["SelfAttention"]
