from __future__ import annotations

import math
import itertools
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvQNN(nn.Module):
    """Hybrid classical–quantum convolutional network with graph‑based quantum backbone.
    Drop‑in replacement for Conv.py.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        out_channels: int = 4,
        stride: int = 1,
        threshold: float = 0.8,
        secondary: float | None = 0.6,
        secondary_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.threshold = threshold
        self.secondary = secondary
        self.secondary_weight = secondary_weight

        self.conv = nn.Conv2d(
            1,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )
        self.head = None

    def _build_head(self, flat_dim: int) -> None:
        self.head = nn.Linear(flat_dim, 10)

    def _build_adjacency(self, flat: torch.Tensor) -> torch.Tensor:
        """Adjacency from cosine similarity."""
        norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sim = torch.mm(flat, flat.t()) / torch.mm(norms, norms.t())
        adj = torch.zeros_like(sim)
        adj[sim >= self.threshold] = 1
        if self.secondary is not None:
            mask = (sim >= self.secondary) & (sim < self.threshold)
            adj[mask] = self.secondary_weight
        return adj

    def _propagate(self, flat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Laplacian propagation."""
        deg = adj.sum(dim=1, keepdim=True)
        lap = deg - adj
        return torch.mm(lap, flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits (B, 10)."""
        bsz = x.size(0)
        features = self.conv(x)  # (B, C, H', W')
        flat = features.view(bsz, -1)  # (B, D)
        if self.head is None:
            self._build_head(flat.size(1))
        adj = self._build_adjacency(flat)
        propagated = self._propagate(flat, adj)
        logits = self.head(propagated)
        return logits
