"""Hybrid attention layer combining classical fully connected and self‑attention.

The class exposes a `run` method that accepts a NumPy array and returns the
output of a linear projection followed by a self‑attention block.  The
implementation uses PyTorch for differentiability and allows end‑to‑end
training on classical hardware.
"""

import torch
from torch import nn
import numpy as np
from typing import Iterable

class HybridAttentionLayerClass(nn.Module):
    """Classical hybrid layer: FC + self‑attention."""

    def __init__(self, n_features: int = 1, embed_dim: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, n_features).

        Returns
        -------
        torch.Tensor
            Output of shape (batch, embed_dim).
        """
        q = self.linear(x)
        k = self.linear(x)
        v = x
        scores = torch.softmax(q @ k.t() / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

    def run(self, x: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper that accepts a NumPy array.
        """
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32)
            return self.forward(tensor).numpy()

def HybridAttentionLayer() -> HybridAttentionLayerClass:
    """Return a ready‑to‑use instance."""
    return HybridAttentionLayerClass()

__all__ = ["HybridAttentionLayer"]
