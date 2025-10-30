import torch
from torch import nn


class QCNNHybrid(nn.Module):
    """A deeper classical convolution‑inspired network with residual connections,
    batch normalization and dropout to match the expressivity of the quantum
    counterpart while remaining fully classical."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        # Residual block 1
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Residual block 2
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Dimensionality reduction analogous to pooling
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        self.head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        residual = x
        x = self.block1(x) + residual
        residual = x
        x = self.block2(x) + residual
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNHybrid:
    """Factory for the upgraded classical QCNN model."""
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "QCNN"]
