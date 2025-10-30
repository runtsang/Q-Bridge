"""Enhanced classical self‑attention with multi‑head support and optional residual connections."""
from __future__ import annotations

import numpy as np
import torch
from typing import Tuple

class SelfAttentionEnhanced:
    """Multi‑head, multi‑layer self‑attention with optional residual connections.

    The interface mimics the original seed: ``run(rotation_params, entangle_params,
    inputs)`` but returns a tuple ``(attention_map, output)``.  The parameters are
    interpreted as linear projections for the query/key/value matrices:

    * ``rotation_params``  – ``n_heads × d_head × d_head``; used for the query
      and value projections.
    * ``entangle_params`` – ``n_heads × d_head × d_head``; used for the key
      projection.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    n_heads : int, default 1
        Number of attention heads.
    n_layers : int, default 1
        Number of stacked attention sub‑blocks.
    residual : bool, default False
        Whether to add a residual connection after each sub‑block.
    """
    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 1,
        n_layers: int = 1,
        residual: bool = False,
    ):
        if embed_dim % n_heads!= 0:
            raise ValueError("embed_dim must be divisible by n_heads")
        self.embed_dim = embed_dim
        self.d_head = embed_dim // n_heads
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.residual = residual

    def _head_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a single head of self‑attention."""
        batch, _ = inputs.shape
        attn_maps = []
        outputs = []
        for h in range(self.n_heads):
            Q = inputs @ rotation_params[h].T          # (batch, d_head)
            K = inputs @ entangle_params[h].T          # (batch, d_head)
            V = inputs @ rotation_params[h].T          # (batch, d_head)
            scores = torch.softmax(
                (Q @ K.T) / np.sqrt(self.d_head), dim=-1
            ).numpy()
            attn_maps.append(scores)
            outputs.append(scores @ V)
        attn_map = np.mean(np.stack(attn_maps), axis=0)
        out = np.mean(np.stack(outputs), axis=0)
        if self.residual:
            out = out + inputs
        return attn_map, out

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute ``n_layers`` stacked attention sub‑blocks.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape ``(n_heads, d_head, d_head)`` – query/value weights.
        entangle_params : np.ndarray
            Shape ``(n_heads, d_head, d_head)`` – key weights.
        inputs : np.ndarray
            Shape ``(batch, embed_dim)``.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (attention_map, output).  ``attention_map`` has shape
            ``(batch, batch)``; ``output`` has shape ``(batch, embed_dim)``.
        """
        out = inputs
        for _ in range(self.n_layers):
            attn_map, out = self._head_attention(rotation_params, entangle_params, out)
        return attn_map, out

__all__ = ["SelfAttentionEnhanced"]
