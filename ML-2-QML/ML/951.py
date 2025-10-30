"""Extended classical QCNN model with residual connections and dropout."""

from __future__ import annotations

import torch
from torch import nn


class QCNNModel(nn.Module):
    """A deeper, regularised QCNN‑inspired network.

    The architecture mirrors the original quantum convolution but adds:
    * Residual connections between consecutive blocks.
    * Dropout after each pooling stage.
    * Optional layer‑wise batch normalisation.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: tuple[int,...] = (16, 16, 12, 8, 4, 4),
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_dims[0]) if use_batchnorm else nn.Identity(),
        )

        self.blocks = nn.ModuleList()
        for i, dim in enumerate(hidden_dims[1:]):
            block = nn.Sequential(
                nn.Linear(hidden_dims[i], dim),
                nn.Tanh(),
                nn.BatchNorm1d(dim) if use_batchnorm else nn.Identity(),
            )
            self.blocks.append(block)

        self.pool = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.feature_map(x)
        for block in self.blocks:
            residual = out
            out = block(out)
            out = out + residual  # residual connection
            out = self.pool(out)
        return torch.sigmoid(self.head(out))


def QCNN() -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
