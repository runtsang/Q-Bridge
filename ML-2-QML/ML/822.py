import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Simple residual block with batch‑norm and dropout."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.residual = nn.Linear(in_features, out_features) if in_features!= out_features else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual
        return F.relu(out)

class QCNNModel(nn.Module):
    """Extended QCNN-inspired architecture with residuals and dropout."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.res1 = ResidualBlock(16, 16)
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.BatchNorm1d(12),
            nn.ReLU()
        )
        self.res2 = ResidualBlock(12, 12)
        self.pool2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        self.res3 = ResidualBlock(8, 8)
        self.pool3 = nn.Sequential(
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU()
        )
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.res1(x)
        x = self.pool1(x)
        x = self.res2(x)
        x = self.pool2(x)
        x = self.res3(x)
        x = self.pool3(x)
        return torch.sigmoid(self.head(x))

def QCNN() -> QCNNModel:
    """Factory returning the configured QCNNModel."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
