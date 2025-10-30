import torch
from torch import nn
import torch.nn.functional as F

class EstimatorQNN(nn.Module):
    """
    A robust regression network with residual connections, dropout,
    and batch normalization. The architecture is an extension of the
    original 2‑input feed‑forward model, adding two hidden layers
    with 32 units each, ReLU activations, and dropout for regularization.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, dropout_rate: float = 0.2):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.residual = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.layer1(x)))
        h = F.relu(self.bn2(self.layer2(h)))
        # Residual skip connection
        h = h + self.residual(h)
        h = self.dropout(h)
        return self.output(h)

__all__ = ["EstimatorQNN"]
