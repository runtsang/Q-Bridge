"""Enhanced QCNN model with residual connections and dropout."""
import torch
from torch import nn

class QCNNGen325(nn.Module):
    """Classical QCNN with skip connections and dropout for improved generalisation."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        # Feature map
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        # Convolutional blocks with residuals
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        # Pooling layers
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        self.pool2 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh()
        )
        # Final head
        self.head = nn.Linear(hidden_dim // 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        # Residual connections
        residual = x
        x = self.conv1(x)
        x = x + residual
        residual = x
        x = self.pool1(x)
        x = self.conv2(x)
        x = x + residual
        residual = x
        x = self.pool2(x)
        x = self.conv3(x)
        x = x + residual
        return torch.sigmoid(self.head(x))

def QCNNGen325Model() -> QCNNGen325:
    """Factory producing a pre‑configured QCNNGen325 instance."""
    return QCNNGen325()

__all__ = ["QCNNGen325", "QCNNGen325Model"]
