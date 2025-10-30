import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """A lightweight residual unit with optional feature‑size adaptation."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )
        self.shortcut = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(x) + self.shortcut(x))

class QCNNModel(nn.Module):
    """
    Classical convolution‑inspired architecture mirroring a QCNN.
    Adds residual connections, batch‑norm and dropout for better generalisation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv1 = ResidualBlock(16, 16)
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv2 = ResidualBlock(12, 8)
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv3 = ResidualBlock(4, 4)
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
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
