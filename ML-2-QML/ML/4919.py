import torch
from torch import nn
import torch.nn.functional as F

class QCNNGen107Model(nn.Module):
    """Hybrid classical QCNN with skip connections, batch‑norm and a regression head."""
    def __init__(self, in_features: int = 8, out_features: int = 1) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.Tanh(),
            nn.BatchNorm1d(16)
        )
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.BatchNorm1d(16)
        )
        self.pool1 = nn.AvgPool1d(2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.BatchNorm1d(8)
        )
        self.pool2 = nn.AvgPool1d(2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.BatchNorm1d(8)
        )
        self.head = nn.Linear(8, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 8)
        x = self.feature_map(x)
        # skip connection 1
        y = self.conv1(x)
        x = x + y
        x = self.pool1(x.unsqueeze(-1)).squeeze(-1)
        # skip connection 2
        y = self.conv2(x)
        x = x + y
        x = self.pool2(x.unsqueeze(-1)).squeeze(-1)
        # skip connection 3
        y = self.conv3(x)
        x = x + y
        out = torch.sigmoid(self.head(x))
        return out

def QCNN() -> QCNNGen107Model:
    """Factory for the hybrid classical QCNN."""
    return QCNNGen107Model()
