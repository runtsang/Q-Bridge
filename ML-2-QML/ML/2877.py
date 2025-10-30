"""
Classical hybrid classifier that mirrors the QCNN structure.
The network first applies a linear feature map, then a series of
convolution‑ and pooling‑like blocks (implemented as linear layers
with tanh activations), and finally a sigmoid‑activated head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.functional import sigmoid

__all__ = ["HybridClassifier"]


class HybridClassifier(nn.Module):
    """Hybrid classical classifier with QCNN‑style architecture."""

    def __init__(self, input_dim: int, depth: int = 3, hidden_sizes: list[int] | None = None) -> None:
        """
        Parameters
        ----------
        input_dim: int
            Dimensionality of the input data.
        depth: int
            Number of convolution‑pooling pairs.
        hidden_sizes: list[int] | None
            Optional explicit list of hidden layer sizes. If None,
            a default progression is used.
        """
        super().__init__()

        if hidden_sizes is None:
            # Default progression: 16 → 16 → 12 → 8 → 4 → 4
            hidden_sizes = [16, 16, 12, 8, 4, 4]

        # Feature map (analogous to ZFeatureMap)
        self.feature_map = nn.Sequential(nn.Linear(input_dim, hidden_sizes[0]), nn.Tanh())

        # Convolution / pooling blocks – all linear layers with tanh
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        idx = 1
        for _ in range(depth):
            self.conv_layers.append(nn.Sequential(nn.Linear(hidden_sizes[idx - 1], hidden_sizes[idx]), nn.Tanh()))
            idx += 1
            self.pool_layers.append(nn.Sequential(nn.Linear(hidden_sizes[idx - 1], hidden_sizes[idx]), nn.Tanh()))
            idx += 1

        # Final classification head
        self.head = nn.Linear(hidden_sizes[-1], 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network.
        Returns probabilities via sigmoid activation.
        """
        x = self.feature_map(x)
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = pool(x)
        return sigmoid(self.head(x))
