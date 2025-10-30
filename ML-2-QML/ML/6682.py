"""Enhanced classical QCNN with metadata and flexible depth.

The model mimics the quantum convolutional architecture but
offers a fully trainable PyTorch implementation.  It exposes
metadata such as weight sizes and encoding positions that
mirror the quantum circuit’s parameter layout, enabling
co‑training or hybrid workflows.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import List, Tuple

class QCNNHybridModel(nn.Module):
    """
    A convolution‑like feed‑forward network that emulates a QCNN.
    Parameters
    ----------
    num_features : int
        Number of input features (must match the quantum feature map).
    depth : int
        Number of convolution‑pooling pairs.
    """

    def __init__(self, num_features: int = 8, depth: int = 3) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth

        # Feature map
        self.feature_map = nn.Sequential(nn.Linear(num_features, 16), nn.Tanh())

        # Build convolution‑pooling stack
        self.layers = nn.ModuleList()
        self.encoding = list(range(num_features))  # metadata: indices of input features
        self.weight_sizes: List[int] = []

        in_dim = 16
        for d in range(depth):
            conv = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Tanh())
            pool = nn.Sequential(nn.Linear(in_dim, in_dim // 2), nn.Tanh())
            self.layers.extend([conv, pool])
            self.weight_sizes.append(sum(p.numel() for p in conv.parameters()))
            self.weight_sizes.append(sum(p.numel() for p in pool.parameters()))
            in_dim = in_dim // 2

        # Final head
        self.head = nn.Linear(in_dim, 1)
        self.weight_sizes.append(sum(p.numel() for p in self.head.parameters()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(self.head(x))

    def get_metadata(self) -> Tuple[List[int], List[int]]:
        """Return encoding positions and weight sizes."""
        return self.encoding, self.weight_sizes

def QCNNHybrid(num_features: int = 8, depth: int = 3) -> QCNNHybridModel:
    """Factory returning the configured :class:`QCNNHybridModel`."""
    return QCNNHybridModel(num_features, depth)

__all__ = ["QCNNHybrid", "QCNNHybridModel"]
