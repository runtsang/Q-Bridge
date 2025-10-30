"""Classical Self‑Attention hybrid module.

This module implements a purely classical self‑attention block
that mirrors the interface of the quantum counterpart.
It can be used as a drop‑in replacement when a quantum backend
is unavailable or for ablation studies.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHybrid(nn.Module):
    """
    Classical self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the key‑value space.
    n_qubits : int, optional
        Number of qubits that the quantum counterpart would use.
        Stored for API compatibility but unused in the classical
        implementation.
    """

    def __init__(self, embed_dim: int, n_qubits: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits

        # Linear projections for query, key and value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute classical self‑attention.

        Parameters
        ----------
        rotation_params : np.ndarray
            Dummy array that matches the quantum interface.
            Ignored in the classical implementation.
        entangle_params : np.ndarray
            Dummy array that matches the quantum interface.
            Ignored in the classical implementation.
        inputs : np.ndarray
            Input matrix of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            Attention‑weighted representation of the inputs.
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Attention scores
        scores = F.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)

        # Weighted sum
        out = scores @ V
        return out.detach().cpu().numpy()


__all__ = ["SelfAttentionHybrid"]
