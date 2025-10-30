"""Hybrid self‑attention module for PyTorch.

This module extends the original seed by adding optional quantum attention
and weight sharing.  The quantum part is implemented in a separate module
(`quantum_attention.py`) and is imported lazily to keep the core module
free of quantum dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

# Lazy import of quantum module
try:
    from.quantum_attention import QuantumAttention
except Exception:
    QuantumAttention = None  # type: ignore

class SelfAttention(nn.Module):
    """Hybrid self‑attention layer.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    use_quantum : bool, default=False
        If True, uses a quantum variational circuit to compute the
        attention weights.  Requires the optional ``quantum_attention``
        module.
    weight_share : bool, default=False
        If True and ``use_quantum=True``, the linear projection weights
        are tied to the rotation parameters of the quantum circuit.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        use_quantum: bool = False,
        weight_share: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_quantum = use_quantum
        self.weight_share = weight_share

        # Linear projections for queries, keys and values
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        if self.use_quantum:
            if QuantumAttention is None:
                raise ImportError(
                    "QuantumAttention module not available. "
                    "Install the optional dependency or set use_quantum=False."
                )
            self.quantum_attention = QuantumAttention(
                embed_dim=embed_dim, weight_share=weight_share
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the self‑attention output.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        # Linear projections
        Q = self.q_proj(x)  # (B, L, D)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Scaled dot‑product scores
        scale = self.embed_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, L, L)

        if self.use_quantum:
            # Flatten batch and seq dimensions for the quantum module
            B, L, _ = x.shape
            flat_scores = scores.reshape(B * L, L).detach().cpu().numpy()
            # Compute quantum attention weights
            quantum_weights = self.quantum_attention.compute(flat_scores)
            # Reshape back to (B, L, L)
            quantum_weights = torch.from_numpy(quantum_weights).to(x.device)
            quantum_weights = quantum_weights.reshape(B, L, L)
            attn = quantum_weights
        else:
            attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)
        return out
