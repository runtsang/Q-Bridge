"""Enhanced classical QCNN-inspired architecture with residual blocks and dropout."""

import torch
from torch import nn

class QCNNPlusModel(nn.Module):
    """QCNN-inspired network with residual connections, batch norm, and dropout."""

    def __init__(self, input_dim: int = 8, hidden_dims: tuple[int,...] = (16, 12, 8, 4)):
        super().__init__()
        self.blocks = nn.ModuleList()
        prev_dim = input_dim
        for dim in hidden_dims:
            block = nn.Sequential(
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
            )
            self.blocks.append(block)
            prev_dim = dim
        self.res_conns = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i]) for i in range(len(hidden_dims))])
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            residual = x
            x = block(x)
            x = x + residual  # residual connection
        x = torch.sigmoid(self.head(x))
        return x

def QCNNplus() -> QCNNPlusModel:
    """Factory for the enhanced QCNN-inspired model."""
    return QCNNPlusModel()

__all__ = ["QCNNplus", "QCNNPlusModel"]
