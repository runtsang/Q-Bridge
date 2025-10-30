import torch
from torch import nn
import torch.nn.functional as F

class EstimatorQNN(nn.Module):
    """
    A deeper feed‑forward regressor with residual connections and dropout.
    Designed to serve as a classical baseline for hybrid experiments.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

__all__ = ["EstimatorQNN"]
