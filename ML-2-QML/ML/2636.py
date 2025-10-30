"""Hybrid fully connected + self‑attention layer for classical training.

The class inherits from nn.Module and implements a forward pass that first
projects the input features with a linear layer and then applies a
single‑head self‑attention block.  The ``run`` method is kept for
compatibility with the original FCL and SelfAttention seeds: it accepts a
sequence of parameters (ignored) and an optional input array.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn


def FCL():
    class HybridLayer(nn.Module):
        def __init__(self, n_features: int = 1, embed_dim: int = 4) -> None:
            super().__init__()
            self.fc = nn.Linear(n_features, embed_dim)
            self.attn = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            proj = self.fc(x)
            attn_out, _ = self.attn(proj, proj, proj)
            return attn_out

        def run(
            self,
            thetas: Iterable[float],
            inputs: Optional[np.ndarray] = None,
        ) -> np.ndarray:
            if inputs is None:
                # Default dummy input: batch 1, seq_len 1
                inputs = np.ones((1, 1, self.fc.in_features))
            x = torch.as_tensor(inputs, dtype=torch.float32)
            out = self.forward(x)
            return out.detach().numpy()

    return HybridLayer()


__all__ = ["FCL"]
