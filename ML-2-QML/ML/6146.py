"""Enhanced classical QCNN with depth‑aware blocks and residual connections."""

from __future__ import annotations

import torch
from torch import nn

class QCNNModelExt(nn.Module):
    """Depth‑aware, modular QCNN‑style network with trainable per‑block scaling.

    Parameters
    ----------
    in_features : int, default 8
        Size of the input vector.
    hidden_size : int, default 16
        Width of each hidden block.
    depth : int, default 3
        Number of convolution‑like blocks.
    """

    def __init__(self, in_features: int = 8, hidden_size: int = 16, depth: int = 3) -> None:
        super().__init__()
        self.depth = depth

        # Feature‑map stage
        self.feature_map = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.Tanh()
        )

        # Build a list of blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(p=0.3)
            )
            self.blocks.append(block)
            # Trainable scale for this block
            setattr(self, f"_scale_{i}", nn.Parameter(torch.ones(1)))

        # Output head
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for i, block in enumerate(self.blocks):
            out = block(x)
            scale = getattr(self, f"_scale_{i}")
            x = x + scale * out   # residual + scaling
        return torch.sigmoid(self.head(x))
