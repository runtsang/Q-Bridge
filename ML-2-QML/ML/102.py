"""Upgraded classical QCNN model with residual connections, batch norm and dropout."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QCNNModel(nn.Module):
    """A deeper, regularised QCNN‑style feed‑forward network."""

    def __init__(self, input_dim: int = 8, hidden_dims: tuple[int,...] = (16, 16, 12, 8, 4, 4), dropout: float = 0.2) -> None:
        """
        Parameters
        ----------
        input_dim: int
            Dimension of the input feature vector.
        hidden_dims: tuple[int,...]
            Layer sizes mirroring the quantum convolution/pooling structure.
        dropout: float
            Dropout probability applied after each block.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for idx, dim in enumerate(hidden_dims):
            # Residual connection after the first two layers
            block = nn.Sequential(
                nn.BatchNorm1d(prev_dim),
                nn.Linear(prev_dim, dim),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
            self.layers.append(block)
            prev_dim = dim

        # Residual addition between layer 1 and layer 3 (if dims match)
        self.residual = nn.Identity() if hidden_dims[0] == hidden_dims[2] else nn.Linear(hidden_dims[0], hidden_dims[2])

        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with a single residual shortcut."""
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i == 1:  # after second block
                res = self.residual(x)
                out = out + res
        return torch.sigmoid(self.head(out))

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for inference."""
        self.eval()
        with torch.no_grad():
            return self.forward(X)

    def initialize_weights(self) -> None:
        """He‑uniform initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="tanh")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


__all__ = ["QCNNModel"]
