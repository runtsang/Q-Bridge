from __future__ import annotations

import numpy as np
import torch
from torch import nn

class UnifiedSelfAttention(nn.Module):
    """
    Hybrid classical self‑attention module.  It combines a standard
    multi‑head attention core with an optional quantum‑inspired
    feature map implemented as random Fourier features.
    """
    def __init__(self,
                 embed_dim: int,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 rbf_dim: int = 256,
                 use_quantum: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        # Classical multi‑head attention
        self.attn = nn.MultiheadAttention(embed_dim, n_heads,
                                          dropout=dropout, batch_first=True)

        # Optional quantum‑inspired feature map
        self.use_quantum = use_quantum
        if use_quantum:
            # Random Fourier feature projection
            self.rbf_proj = nn.Parameter(
                torch.randn(embed_dim, rbf_dim) * 0.1, requires_grad=False
            )
            self.rbf_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # Linear readout (FCL style)
        readout_input_dim = embed_dim + (rbf_dim if use_quantum else 0)
        self.readout = nn.Linear(readout_input_dim, embed_dim)

    def _rbf_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Random Fourier feature map: z = sqrt(2/d) * cos(xW + b)
        """
        z = torch.matmul(x, self.rbf_proj) * self.rbf_scale
        return torch.cos(z) * np.sqrt(2.0 / self.rbf_proj.shape[1])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input embeddings of shape (B, T, D).
        mask : torch.Tensor or None
            Optional padding mask.
        """
        # Classical attention
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        attn_output = self.dropout(attn_output)

        if self.use_quantum:
            q_features = self._rbf_map(x)
            combined = torch.cat([attn_output, q_features], dim=-1)
        else:
            combined = attn_output

        out = self.readout(combined)
        return out

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Compatibility wrapper that accepts the original seed signature.
        The quantum parameters are ignored in this classical implementation.
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)
        out = self.forward(x)
        return out.detach().cpu().numpy()

__all__ = ["UnifiedSelfAttention"]
