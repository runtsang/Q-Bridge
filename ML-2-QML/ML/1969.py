import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block with two linear layers, batch‑norm and dropout."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.bn1(x)
        out = F.relu(out)
        out = self.linear1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = self.dropout(out)
        if residual.shape[-1]!= out.shape[-1]:
            residual = nn.Linear(residual.shape[-1], out.shape[-1]).to(out.device)(residual)
        return out + residual

class QCNNModel(nn.Module):
    """ResNet‑style QCNN with dropout and batch‑norm."""
    def __init__(self,
                 input_dim: int = 8,
                 hidden_dims: list[int] | None = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [16, 12, 8, 4]
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(ResidualBlock(in_dim, h, dropout=dropout))
            in_dim = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        x = self.head(x)
        return torch.sigmoid(x)

def QCNN() -> QCNNModel:
    """Factory returning a fully‑initialized QCNNModel."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
