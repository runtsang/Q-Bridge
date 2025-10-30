import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Linear layer followed by a Tanh activation."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.lin(x))

class ResidualBlock(nn.Module):
    """Adds a skip connection around a ConvBlock."""
    def __init__(self, features: int) -> None:
        super().__init__()
        self.block = ConvBlock(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class QCNN(nn.Module):
    """Extended QCNN with residual connections, variable depth, and dropout."""
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 12, 8, 4, 4]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(ConvBlock(prev, h))
            layers.append(ResidualBlock(h))
            prev = h
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.dropout(x)
        return torch.sigmoid(self.head(x))

def QCNN() -> QCNN:
    """Factory returning a ready‑to‑train QCNN instance."""
    return QCNN()

__all__ = ["QCNN", "QCNNModel"]  # keep QCNNModel for backward compatibility
