import torch
from torch import nn
import torch.nn.functional as F

class QCNNHybrid(nn.Module):
    """
    A deeper, residual‑enhanced QCNN‑style network for classical data.
    The architecture mirrors the original 8‑to‑4‑to‑2‑to‑1 transition but adds
    skip connections, batch‑normalisation and dropout to improve generalisation.
    """
    def __init__(self, in_features: int = 8, hidden_features: int = 16) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU()
        )
        # First convolution block with residual
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU()
        )
        self.res1 = nn.Identity()
        # Pooling block with dropout
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_features, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        # Residual connection
        self.res2 = nn.Identity()
        # Pooling block
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        # Third convolution
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
            nn.ReLU()
        )
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = x + self.res1(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = x + self.res2(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        logits = self.head(x)
        return torch.sigmoid(logits)

def QCNNHybrid() -> QCNNHybrid:
    """
    Factory returning a fully configured QCNNHybrid model.
    """
    return QCNNHybrid()

__all__ = ["QCNNHybrid"]
