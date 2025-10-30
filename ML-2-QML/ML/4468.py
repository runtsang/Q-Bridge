"""Hybrid self‑attention module with classical implementations.

The class bundles a classical attention block, a sampler network, a fully‑connected
layer and a regression estimator.  It preserves the original API while
expanding functionality for research experiments.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

class HybridSelfAttention:
    """
    Classical hybrid self‑attention that integrates four reference modules:
    - Classical self‑attention (SelfAttention)
    - Classical sampler network (SamplerQNN)
    - Fully‑connected layer (FCL)
    - Estimator regression network (EstimatorQNN)
    """

    def __init__(self, embed_dim: int = 4, n_features: int = 1) -> None:
        self.embed_dim = embed_dim

        # Random rotation and entangle parameters for attention
        self.rotation_params = np.random.randn(embed_dim * 3).astype(np.float32)
        self.entangle_params = np.random.randn(embed_dim - 1).astype(np.float32)

        # Attention module
        self.attention = self._build_attention()

        # Sampler network
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

        # Fully‑connected layer
        self.fcl = nn.Linear(n_features, 1)

        # Estimator network
        self.estimator = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    # ---------- Classical components ----------
    def _build_attention(self) -> nn.Module:
        class Attention(nn.Module):
            def __init__(self, embed_dim: int, rotation_params: np.ndarray,
                         entangle_params: np.ndarray) -> None:
                super().__init__()
                self.embed_dim = embed_dim
                self.rotation_params = torch.as_tensor(
                    rotation_params.reshape(embed_dim, -1), dtype=torch.float32
                )
                self.entangle_params = torch.as_tensor(
                    entangle_params.reshape(embed_dim, -1), dtype=torch.float32
                )

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                query = inputs @ self.rotation_params
                key = inputs @ self.entangle_params
                scores = F.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
                return scores @ inputs

        return Attention(self.embed_dim, self.rotation_params, self.entangle_params)

    def run_attention(self, inputs: np.ndarray) -> np.ndarray:
        """Compute the classical self‑attention output."""
        inp = torch.as_tensor(inputs, dtype=torch.float32)
        out = self.attention(inp)
        return out.detach().numpy()

    def run_sampler(self, inputs: np.ndarray) -> np.ndarray:
        """Sample a probability distribution with the classical sampler."""
        inp = torch.as_tensor(inputs, dtype=torch.float32)
        probs = F.softmax(self.sampler(inp), dim=-1)
        return probs.detach().numpy()

    def run_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        """Apply the fully‑connected layer to a sequence of scalars."""
        vals = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = torch.tanh(self.fcl(vals)).mean(dim=0)
        return out.detach().numpy()

    def run_estimator(self, inputs: np.ndarray) -> np.ndarray:
        """Regress a scalar from a 2‑dimensional input."""
        inp = torch.as_tensor(inputs, dtype=torch.float32)
        out = self.estimator(inp)
        return out.detach().numpy()

    # ---------- Combined interface ----------
    def run(self, inputs: np.ndarray) -> dict:
        """
        Return a dictionary containing the outputs of all four components.
        """
        return {
            "attention": self.run_attention(inputs),
            "sampler": self.run_sampler(inputs),
            "fcl": self.run_fcl([0.0, 1.0, 2.0]),
            "estimator": self.run_estimator(inputs),
        }

def SelfAttention() -> HybridSelfAttention:
    """Factory that returns a HybridSelfAttention instance with default settings."""
    return HybridSelfAttention(embed_dim=4)

__all__ = ["SelfAttention"]
