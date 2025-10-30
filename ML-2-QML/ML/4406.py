"""Hybrid classical self‑attention with noise simulation.

The class mirrors the quantum interface: ``run(rotation_params,
entangle_params, inputs)``.  ``rotation_params`` and ``entangle_params``
are optional weight matrices that can overwrite the internal linear
layers.  When ``noise_shots`` is provided, Gaussian shot noise is added
to the output to emulate realistic quantum measurements.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSelfAttention(nn.Module):
    """Multi‑head self‑attention with optional Gaussian shot noise.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability in the attention block.
    noise_shots : int | None, default None
        If set, Gaussian noise with variance 1/noise_shots is added to
        the output to emulate quantum shot noise.
    noise_seed : int | None, default None
        Random seed for reproducible noise.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        noise_shots: int | None = None,
        noise_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.noise_shots = noise_shots
        self.noise_seed = noise_seed
        self.rng = np.random.default_rng(noise_seed)

        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def _apply_params(
        self,
        rotation_params: np.ndarray | None,
        entangle_params: np.ndarray | None,
    ) -> None:
        """Inject external parameters into the attention projection weights."""
        if rotation_params is not None:
            # Expect shape (embed_dim, embed_dim)
            self.attn.in_proj_weight.data = torch.from_numpy(
                rotation_params
            ).float()
        if entangle_params is not None:
            # Scale key and value projections to emulate entanglement effects
            scale = torch.from_numpy(entangle_params).float()
            # Key projection is the second block of weights
            self.attn.in_proj_weight.data[
                :, self.embed_dim : 2 * self.embed_dim
            ] *= scale
            # Value projection is the third block of weights
            self.attn.in_proj_weight.data[
                :, 2 * self.embed_dim :
            ] *= scale

    def run(
        self,
        rotation_params: np.ndarray | None,
        entangle_params: np.ndarray | None,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Forward pass with optional parameter injection and shot noise.

        Parameters
        ----------
        rotation_params : np.ndarray | None
            Optional matrix of shape (embed_dim, embed_dim) to overwrite
            the query/key/value projections.
        entangle_params : np.ndarray | None
            Optional scaling vector for key/value projections.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Attention‑weighted output of shape (batch, seq_len, embed_dim).
        """
        batch = torch.as_tensor(inputs, dtype=torch.float32)
        self._apply_params(rotation_params, entangle_params)
        attn_output, _ = self.attn(batch, batch, batch)
        out = self.drop(self.norm(attn_output))

        if self.noise_shots is not None:
            noise_std = 1.0 / np.sqrt(self.noise_shots)
            noise = self.rng.normal(
                0.0, noise_std, size=out.shape
            ).astype(np.float32)
            out = out + torch.from_numpy(noise)

        return out.numpy()


def SelfAttention() -> HybridSelfAttention:
    """Return a default hybrid self‑attention instance."""
    return HybridSelfAttention(embed_dim=4)
