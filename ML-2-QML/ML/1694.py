"""Enhanced classical convolution‑inspired network.

This version introduces residual skip connections, batch‑normalization,
and dropout to improve expressivity and regularisation while keeping
the overall layer sequence similar to the original QCNN design.
"""

import torch
from torch import nn

class QCNNEnhanced(nn.Module):
    """Fully‑connected network with residual blocks and dropout."""
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: tuple[int,...] = (16, 12, 8, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList()
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.blocks.append(block)
        self.head = nn.Linear(hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        for block in self.blocks:
            # Residual connection: dimensions match because each block
            # maps a vector to the same size as its input.
            residual = x
            x = block(x)
            x = x + residual
        return self.sigmoid(self.head(x))

def QCNNEnhanced() -> QCNNEnhanced:
    """Factory returning a configured :class:`QCNNEnhanced`."""
    return QCNNEnhanced()

__all__ = ["QCNNEnhanced", "QCNNEnhanced"]
