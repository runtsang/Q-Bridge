"""
Self‑attention helper with optional quantum sampler integration.
The class mimics the original SelfAttention interface but
includes an optional `sampler` argument that can be any callable
returning a probability vector (e.g., a Qiskit SamplerQNN).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["SelfAttention"]


class SelfAttention:
    """Hybrid classical self‑attention module.

    Parameters
    ----------
    embed_dim : int
        Size of the embedding (dimensionality of Q, K, V).
    sampler : Callable[[torch.Tensor], torch.Tensor] | None
        Optional callable that receives a tensor of shape (N, embed_dim)
        and returns a probability distribution per sample.  If provided,
        it overrides the softmax used in the classical branch.
    """

    def __init__(self, embed_dim: int, sampler: None | torch.Tensor = None) -> None:
        self.embed_dim = embed_dim
        self.sampler = sampler

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute self‑attention using either classical softmax or a quantum sampler.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (embed_dim, 3) used only by the quantum branch.
        entangle_params : np.ndarray
            Shape (embed_dim-1,) used only by the quantum branch.
        inputs : np.ndarray
            Input matrix of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            Attention‑weighted values of shape (batch, embed_dim).
        """
        # Classical branch
        queries = torch.tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        keys = torch.tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        values = torch.tensor(inputs, dtype=torch.float32)

        scores = torch.softmax(queries @ keys.T / np.sqrt(self.embed_dim), dim=-1)

        # Override with sampler if provided
        if self.sampler is not None:
            # The sampler is expected to return a (batch, embed_dim) distribution
            scores = self.sampler(scores)

        return (scores @ values).numpy()
