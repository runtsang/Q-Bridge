import torch
from torch import nn
import torch.nn.functional as F

class QCNNHybrid(nn.Module):
    """A deep residual QCNN-inspired network with batch‑norm and dropout."""
    def __init__(self, input_dim: int = 8, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            # Residual block: linear → tanh → batchnorm → dropout
            block = nn.Sequential(
                nn.Linear(prev_dim, dim),
                nn.Tanh(),
                nn.BatchNorm1d(dim),
                nn.Dropout(p=0.1)
            )
            layers.append(block)
            prev_dim = dim
        self.blocks = nn.ModuleList(layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            residual = x
            x = block(x)
            # Add residual only if dimensions match
            if residual.shape[-1] == x.shape[-1]:
                x = x + residual
        out = self.head(x)
        return torch.sigmoid(out)

def QCNNHybridFactory() -> QCNNHybrid:
    """Convenience factory returning a ready‑to‑train QCNNHybrid."""
    return QCNNHybrid()

__all__ = ["QCNNHybrid", "QCNNHybridFactory"]
