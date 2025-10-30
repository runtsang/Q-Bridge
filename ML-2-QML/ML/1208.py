import torch
from torch import nn
import torch.nn.functional as F

class QCNNModel(nn.Module):
    """Enhanced QCNN-inspired classical network with residual connections, batch‑norm, and dropout."""
    def __init__(self, input_dim: int = 8, hidden_dims: list = None, dropout: float = 0.2) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh()
        )
        # Convolutional layers with residual skip
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.BatchNorm1d(hidden_dims[3]),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.pool2 = nn.Sequential(
            nn.Linear(hidden_dims[3], hidden_dims[4]),
            nn.BatchNorm1d(hidden_dims[4]),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Linear(hidden_dims[4], hidden_dims[5]),
            nn.BatchNorm1d(hidden_dims[5]),
            nn.Tanh()
        )
        self.head = nn.Linear(hidden_dims[5], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

def QCNN() -> QCNNModel:
    """Factory returning a fully‑configured QCNNModel."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
