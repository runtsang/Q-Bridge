"""Hybrid Classical Self‑Attention module with trainable projection and optional masking."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Classical self‑attention block that can be used as a drop‑in replacement
    for the original ``SelfAttention`` helper.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    out_dim : int | None, optional
        If provided, a linear projection is applied to the attention output.
        Defaults to ``None`` (no projection).
    mask : bool, optional
        When ``True`` a causal mask (upper‑triangular) is applied to the
        attention scores, useful for autoregressive models.
    """

    def __init__(
        self,
        embed_dim: int,
        out_dim: int | None = None,
        mask: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.mask = mask
        if out_dim is not None:
            self.proj = nn.Linear(embed_dim, out_dim)

    def _compute_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scaled dot‑product attention scores."""
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        if self.mask:
            # Causal mask: zero out lower‑triangular part
            mask_tensor = torch.triu(torch.ones_like(scores), diagonal=1)
            scores = scores.masked_fill(mask_tensor == 0, float("-inf"))
        return F.softmax(scores, dim=-1)

    def forward(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """
        Forward pass of the self‑attention block.

        Parameters
        ----------
        inputs : np.ndarray
            Input embeddings of shape (batch, seq_len, embed_dim).
        rotation_params : np.ndarray
            Parameters used to generate the query matrix. Expected shape
            (embed_dim, embed_dim).
        entangle_params : np.ndarray
            Parameters used to generate the key matrix. Expected shape
            (embed_dim, embed_dim).

        Returns
        -------
        np.ndarray
            Output embeddings of shape (batch, seq_len, out_dim) if
            ``out_dim`` is set, otherwise (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = inputs.shape
        x = torch.as_tensor(inputs, dtype=torch.float32)

        # Compute query, key, value
        query = torch.matmul(x, torch.as_tensor(rotation_params, dtype=torch.float32))
        key   = torch.matmul(x, torch.as_tensor(entangle_params, dtype=torch.float32))
        value = x

        # Attention scores and weighted sum
        scores = self._compute_scores(query, key)
        attn_out = torch.matmul(scores, value)

        # Optional projection
        if self.out_dim is not None:
            attn_out = self.proj(attn_out)

        return attn_out.detach().cpu().numpy()

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Compatibility wrapper matching the original interface."""
        return self.forward(inputs, rotation_params, entangle_params)

__all__ = ["SelfAttention"]
