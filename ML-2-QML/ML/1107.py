import torch
from torch import nn
import torch.nn.functional as F

class QCNNHybrid(nn.Module):
    """Extended QCNN‑inspired classical network with residuals, dropout, and batch‑norm."""
    def __init__(self, input_dim: int = 8, hidden_dims: list[int] | tuple[int,...] = (16, 16, 12, 8, 4, 4), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.layers(x)
        # residual connection from input to final hidden layer
        if residual.shape[-1] == out.shape[-1]:
            out = out + residual
        out = self.head(out)
        return torch.sigmoid(out)

def QCNN() -> QCNNHybrid:
    """Factory returning the configured :class:`QCNNHybrid`."""
    return QCNNHybrid()

__all__ = ["QCNN", "QCNNHybrid"]
