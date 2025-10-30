"""Enhanced classical QCNN model with residual connections and dropout.

This module defines a feed‑forward network that emulates the structure of a
quantum convolutional neural network.  Each convolution‑pooling block
consists of a linear layer, a tanh activation, a dropout layer and a
second linear layer.  A residual connection adds the block input to its
output, improving gradient flow and regularisation.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QCNNEnhancedModel(nn.Module):
    """Feed‑forward network that mimics a quantum convolutional neural network.

    Parameters
    ----------
    input_dim : int, default 8
        Size of the input vector.
    num_blocks : int, default 3
        Number of convolution‑pooling blocks.
    dropout : float, default 0.2
        Dropout probability applied within each block.
    """

    def __init__(self, input_dim: int = 8, num_blocks: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh()
        )
        # Build a stack of blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(16, 16),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(16, 12),
                nn.Tanh()
            )
            self.blocks.append(block)
        # Final classifier
        self.final = nn.Sequential(
            nn.Linear(12, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual   # residual connection
            x = torch.relu(x)
        x = self.final(x)
        return torch.sigmoid(x)


def QCNNEnhanced(backend: str = "cpu") -> nn.Module:
    """Return a QCNNEnhanced model.

    Parameters
    ----------
    backend : str, optional
        'cpu' for the classical model.  The quantum backend is not
        available from this module and will raise a ``ValueError``.
    """
    if backend.lower() == "cpu":
        return QCNNEnhancedModel()
    raise ValueError(f"Unsupported backend '{backend}'. Only 'cpu' is supported in the classical module.")
