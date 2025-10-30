"""
HybridEstimatorQNN: A deep feed‑forward regressor with residual connections, dropout,
and batch normalization. Designed to be compatible with the QML counterpart for
hybrid training experiments.
"""

import torch
from torch import nn

class HybridEstimatorQNN(nn.Module):
    """
    A more expressive regression network that includes:
    - Two hidden layers with 32 units each.
    - Residual connections between layers.
    - Dropout (p=0.2) to mitigate overfitting.
    - Batch normalization after each linear layer.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32,
                 output_dim: int = 1, dropout: float = 0.2) -> None:
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
            nn.Linear(hidden_dim, output_dim),
        )
        self.residual = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        # Residual connection
        res = self.residual(x)
        return out + res

def EstimatorQNN() -> HybridEstimatorQNN:
    """Factory returning an instance of the enhanced estimator."""
    return HybridEstimatorQNN()

__all__ = ["HybridEstimatorQNN", "EstimatorQNN"]
