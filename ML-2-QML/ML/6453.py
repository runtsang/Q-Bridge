import torch
from torch import nn
import torch.nn.functional as F

class QCNNBlock(nn.Module):
    """Single convolution‑like layer with optional batch‑norm and dropout."""
    def __init__(self, in_features: int, out_features: int,
                 dropout: float = 0.0, batch_norm: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        x = F.tanh(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class QCNNEnhancedModel(nn.Module):
    """Modular QCNN‑style network that supports configurable depth and regularisation."""
    def __init__(self,
                 input_dim: int = 8,
                 hidden_sizes: list[int] | None = None,
                 dropout: float = 0.0,
                 batch_norm: bool = False) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [16, 16, 12, 8, 4]
        self.blocks = nn.ModuleList()
        in_dim = input_dim
        for out_dim in hidden_sizes:
            self.blocks.append(QCNNBlock(in_dim, out_dim,
                                         dropout=dropout,
                                         batch_norm=batch_norm))
            in_dim = out_dim
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return torch.sigmoid(self.head(x))

def QCNNEnhanced() -> QCNNEnhancedModel:
    """Factory returning a default‑configured :class:`QCNNEnhancedModel`."""
    return QCNNEnhancedModel()

__all__ = ["QCNNEnhanced", "QCNNEnhancedModel", "QCNNBlock"]
