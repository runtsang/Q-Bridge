"""Hybrid classical estimator that outputs quantum rotation angles.

The network mirrors the QFCModel CNN backbone, adds a self‑attention
block, and maps the attended representation to four parameters that
will be fed into a variational quantum circuit."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleSelfAttention(nn.Module):
    """A minimal self‑attention block operating on a flat feature vector."""
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key   = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        return torch.matmul(scores, v)

class HybridEstimator(nn.Module):
    """
    Classical network that outputs four rotation angles for a quantum circuit.

    Architecture
    ------------
    * Conv2d → ReLU → MaxPool2d → Conv2d → ReLU → MaxPool2d
    * Flatten → Self‑attention (SimpleSelfAttention)
    * Linear → ReLU → Linear → Tanh → BatchNorm1d
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flattened feature size after two 2×2 poolings on 28×28 input
        self.flattened = 16 * 7 * 7
        self.attention = SimpleSelfAttention(self.flattened)
        self.head = nn.Sequential(
            nn.Linear(self.flattened, 64),
            nn.ReLU(),
            nn.Linear(64, 4),          # output 4 rotation angles
            nn.Tanh(),                 # constrain to [-1, 1] for angles
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of images of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Batch of shape (B, 4) containing rotation angles in radians.
        """
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        attn_out = self.attention(flat)
        out = self.head(attn_out)
        return self.norm(out)

__all__ = ["HybridEstimator"]
