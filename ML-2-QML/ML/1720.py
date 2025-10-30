"""Hybrid-inspired classical QCNN with residual blocks and layer normalization.

The model extends the original fully‑connected stack with:
* Residual connections that mirror the pooling stages of the quantum circuit.
* LayerNorm after each non‑linear block to stabilise training.
* Optional dropout for regularisation.

This makes the network more expressive while keeping the original
convolution‑like intuition.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class QCNN(nn.Module):
    """Classical convolution‑inspired network with residual connections.

    Parameters
    ----------
    in_features : int, optional
        Size of the input vector.  Default 8.
    hidden_sizes : Sequence[int], optional
        Sizes of the hidden layers.  Default (16, 16, 12, 8, 4, 4).
    dropout : float, optional
        Dropout probability.  Default 0.0.
    """

    def __init__(
        self,
        in_features: int = 8,
        hidden_sizes: tuple[int,...] = (16, 16, 12, 8, 4, 4),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Build layers with residual links
        self.layers = nn.ModuleList()
        prev = in_features
        for size in hidden_sizes:
            block = nn.Sequential(
                nn.Linear(prev, size),
                nn.LayerNorm(size),
                nn.Tanh(),
                self.dropout,
            )
            self.layers.append(block)
            prev = size

        # Residual connections after pooling‑like reductions
        self.residual = nn.ModuleDict(
            {
                "pool1": nn.Linear(16, 12),
                "pool2": nn.Linear(8, 4),
                "pool3": nn.Linear(4, 4),
            }
        )

        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial feature map
        x = self.layers[0](x)

        # Residual 1
        res = self.residual["pool1"](x)
        x = self.layers[1](x) + res

        # Residual 2
        res = self.residual["pool2"](x)
        x = self.layers[2](x) + res

        # Residual 3
        res = self.residual["pool3"](x)
        x = self.layers[3](x) + res

        # Remaining layers
        for layer in self.layers[4:]:
            x = layer(x)

        return torch.sigmoid(self.head(x))


def QCNN(
    in_features: int = 8,
    hidden_sizes: tuple[int,...] | None = None,
    dropout: float = 0.0,
) -> QCNN:
    """Factory returning a configured :class:`QCNN`."""
    if hidden_sizes is None:
        hidden_sizes = (16, 16, 12, 8, 4, 4)
    return QCNN(in_features, hidden_sizes, dropout)


__all__ = ["QCNN"]
