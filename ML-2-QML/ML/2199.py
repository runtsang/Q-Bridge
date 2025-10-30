"""Extended classical QCNN with residual connections and dropout.

The architecture keeps the original block structure but adds skip connections,
dropout regularisation and a configurable hidden size.  This yields a richer
representation while remaining fully classical.
"""

from __future__ import annotations

import torch
from torch import nn

class QCNNModel(nn.Module):
    """Residual convolution‑pooling network with dropout.

    Parameters
    ----------
    input_dim : int, default 8
        Dimension of the input feature vector.
    hidden_dims : list[int], default [16, 12, 8]
        Hidden sizes for each residual block.
    dropout : float, default 0.2
        Drop‑out probability applied after each pooling stage.
    """

    def __init__(self, input_dim: int = 8, hidden_dims: list[int] | tuple[int,...] = (16, 12, 8), dropout: float = 0.2) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), nn.ReLU())
        self.blocks = nn.ModuleList()
        for hd in hidden_dims:
            conv = nn.Sequential(nn.Linear(hd, hd), nn.ReLU())
            pool = nn.Sequential(nn.Linear(hd, hd // 2), nn.ReLU())
            drop = nn.Dropout(dropout)
            self.blocks.append(nn.ModuleDict({"conv": conv, "pool": pool, "dropout": drop}))
        self.head = nn.Linear(hidden_dims[-1] // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for block in self.blocks:
            y = block["conv"](x)
            y = block["pool"](y)
            y = block["dropout"](y)
            x = x + y  # residual skip connection
        return torch.sigmoid(self.head(x))

def QCNN() -> QCNNModel:
    """Convenience factory mirroring the original seed."""
    return QCNNModel()

__all__ = ["QCNNModel", "QCNN"]
