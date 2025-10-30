"""Hybrid self‑attention module that fuses classical attention with a quantum sampler.

The class exposes a convenient API that mirrors the original SelfAttention
and SamplerQNN interfaces while integrating their strengths.  The
`run` method accepts both classical rotation/entangle parameters and
quantum sampler weights, returning a weighted combination of the two
outputs.  This design allows experiments that compare pure classical,
pure quantum, and hybrid scaling on the same data pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalSelfAttention:
    """Simple dot‑product self‑attention implemented in PyTorch."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> torch.Tensor:
        # Reshape parameters to match embedding dimensions
        rot = torch.tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        ent = torch.tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        inp = torch.tensor(inputs, dtype=torch.float32)

        query = inp @ rot
        key = inp @ ent
        value = inp

        scores = F.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value


class SamplerModule(nn.Module):
    """Feed‑forward sampler that outputs a probability distribution."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)


class HybridAttentionSampler:
    """Hybrid module combining classical attention and quantum‑style sampling."""

    def __init__(self, embed_dim: int = 4):
        self.classical = ClassicalSelfAttention(embed_dim)
        self.sampler = SamplerModule()

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        sampler_inputs: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        inputs : np.ndarray
            Input features for the attention block.
        rotation_params, entangle_params : np.ndarray
            Parameters for the classical attention sub‑module.
        sampler_inputs : np.ndarray
            2‑dimensional input to the sampler (e.g., two‑hot encoded tokens).
        alpha : float, default 0.5
            Weighting factor between classical and quantum outputs.

        Returns
        -------
        np.ndarray
            Combined attention distribution.
        """
        # Classical attention
        attn = self.classical.forward(rotation_params, entangle_params, inputs)

        # Quantum sampler output
        samp = self.sampler(torch.tensor(sampler_inputs, dtype=torch.float32)).numpy()

        # Blend the two distributions
        blended = alpha * attn + (1 - alpha) * samp
        # Normalize to ensure a proper probability distribution
        return blended / blended.sum(axis=-1, keepdims=True)

__all__ = ["HybridAttentionSampler"]
