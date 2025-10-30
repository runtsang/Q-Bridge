"""Hybrid classical self‑attention with fully‑connected preprocessing.

The module defines a SelfAttention() factory that returns a
ClassicalHybridSelfAttention instance.  It mirrors the public API of the
original SelfAttention class while adding a fully‑connected layer
pre‑processing step.  The run method accepts rotation and entanglement
parameter vectors, a list of FC‑layer thetas, and an input matrix.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable

def SelfAttention():
    class ClassicalHybridSelfAttention(nn.Module):
        """Classical self‑attention with a fully‑connected preprocessing layer."""

        def __init__(self, embed_dim: int = 4, n_features: int = 1):
            super().__init__()
            self.embed_dim = embed_dim
            self.fc = nn.Linear(n_features, embed_dim)
            self.linear = nn.Linear(embed_dim, embed_dim)

        def run(
            self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            fcl_thetas: Iterable[float],
            inputs: np.ndarray,
        ) -> np.ndarray:
            # Fully‑connected preprocessing
            x = torch.as_tensor(inputs, dtype=torch.float32)
            x_fc = self.fc(x)

            # Build query & key matrices
            rot = torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
            ent = torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
            queries = torch.matmul(x_fc, rot)
            keys = torch.matmul(x_fc, ent)

            # Attention scores
            scores = torch.softmax(queries @ keys.T / np.sqrt(self.embed_dim), dim=-1)

            # Value transformation
            values = torch.tanh(self.linear(x_fc))

            # Weighted sum
            out = scores @ values
            return out.detach().numpy()

    return ClassicalHybridSelfAttention()
