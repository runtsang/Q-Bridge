"""Enhanced classical QCNN with residual connections and dropout."""
import torch
from torch import nn

class QCNNEnhanced(nn.Module):
    """A deeper QCNN‑inspired network with residuals, batch norm and dropout."""
    def __init__(self, input_dim: int = 8, num_classes: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        # Residual block 1
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        # Residual block 2
        self.conv2 = nn.Sequential(
            nn.Linear(16, 12),
            nn.BatchNorm1d(12),
            nn.ReLU()
        )
        # Residual block 3
        self.conv3 = nn.Sequential(
            nn.Linear(12, 8),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        self.pool = nn.Sequential(
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head = nn.Linear(4, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        # block 1
        residual = x
        x = self.conv1(x)
        x = x + residual
        # block 2
        residual = x
        x = self.conv2(x)
        x = x + residual
        # block 3
        residual = x
        x = self.conv3(x)
        x = x + residual
        x = self.pool(x)
        logits = self.head(x)
        return self.sigmoid(logits)

def QCNNEnhanced() -> QCNNEnhanced:
    """Factory returning the configured QCNNEnhanced model."""
    return QCNNEnhanced()

__all__ = ["QCNNEnhanced"]
