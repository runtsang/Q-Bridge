"""Classical self-attention with regression head, inspired by SelfAttention.py and EstimatorQNN.py."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SelfAttentionEstimator:
    """
    Combines a classical self‑attention block with a small feed‑forward regressor.
    The attention mechanism is identical to the one in the original SelfAttention seed,
    but its output is passed through a 2‑layer MLP to produce a scalar prediction.
    """

    def __init__(self, embed_dim: int = 4, hidden_dim: int = 8) -> None:
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute self‑attention scores and feed the result through the regressor.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the query and key projections (shape: (embed_dim,)).
        entangle_params : np.ndarray
            Parameters for the value projection (shape: (embed_dim,)).
        inputs : np.ndarray
            Input features (shape: (batch, embed_dim)).

        Returns
        -------
        np.ndarray
            Regression output (shape: (batch, 1)).
        """
        # --- Self‑attention ---
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        attn_out = scores @ value

        # --- Regression ---
        out = self.regressor(attn_out)
        return out.detach().numpy()


__all__ = ["SelfAttentionEstimator"]
