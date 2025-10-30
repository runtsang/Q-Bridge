import torch
from torch import nn
import torch.nn.functional as F

class QCNNModel(nn.Module):
    """
    A scalable, residual QCNN‑inspired architecture.

    The network emulates a quantum convolutional neural network with
    * feature mapping (linear + activation)
    * multiple residual convolutional blocks (linear + activation)
    * global average pooling
    * dropout and batch norm for regularisation
    * final classification head with sigmoid activation

    The module is fully differentiable and can be trained with any
    PyTorch optimiser. It accepts input tensors of shape (batch, 8)
    but can be easily extended to other dimensionalities.
    """

    def __init__(self, in_features: int = 8, hidden_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh()
        )
        # Residual convolutional blocks
        self.conv_blocks = nn.ModuleList([
            self._residual_block(hidden_dim),
            self._residual_block(hidden_dim),
            self._residual_block(hidden_dim // 2)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim // 2, 1)

    def _residual_block(self, dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for block in self.conv_blocks:
            residual = x
            x = block(x)
            x = x + residual  # skip connection
        x = self.pool(x.unsqueeze(-1)).squeeze(-1)
        x = self.dropout(x)
        return torch.sigmoid(self.head(x))

def QCNN() -> QCNNModel:
    """Factory that returns a pre‑configured QCNNModel."""
    return QCNNModel()
