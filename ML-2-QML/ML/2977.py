"""Classical self‑attention module with an RBF kernel for score computation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

class RBFKernel(nn.Module):
    """Radial‑basis‑function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)          # [B, B, D]
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))  # [B, B]

class ClassicalSelfAttention(nn.Module):
    """Self‑attention that uses a kernel‑based similarity instead of dot‑product."""
    def __init__(self, embed_dim: int = 4, gamma: float = 1.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel = RBFKernel(gamma)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute attention weighted outputs.

        Parameters
        ----------
        rotation_params : ndarray
            Parameters for linear transformations of the query.
        entangle_params : ndarray
            Parameters for linear transformations of the key.
        inputs : ndarray
            Input feature matrix [batch, features].

        Returns
        -------
        ndarray
            Attention‑weighted output [batch, features].
        """
        # Linear projections
        query = torch.tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.tensor(inputs, dtype=torch.float32)

        # Kernel‑based similarity
        scores = self.kernel(query, key)                    # [B, B]
        scores = scores / torch.sum(scores, dim=-1, keepdim=True)  # normalise

        return (scores @ value).numpy()

def SelfAttention() -> ClassicalSelfAttention:
    """Factory providing a ready‑to‑use classical self‑attention instance."""
    return ClassicalSelfAttention()

__all__ = ["SelfAttention", "ClassicalSelfAttention", "RBFKernel"]
