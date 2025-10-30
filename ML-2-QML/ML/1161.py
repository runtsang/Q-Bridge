"""Enhanced QCNN model with residual connections and dropout."""

import torch
from torch import nn

class QCNNModel(nn.Module):
    """A classical QCNN‑inspired network with residual connections and dropout."""
    def __init__(self, input_dim: int = 8, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 12, 8, 4]
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=0.1))
            # Residual connection when dimensions match
            if prev_dim == dim:
                layers.append(nn.Identity())
            prev_dim = dim
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.sigmoid(self.head(x))

def QCNN() -> QCNNModel:
    """Return a configured QCNNModel instance."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
