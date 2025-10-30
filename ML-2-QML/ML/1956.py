"""Enhanced feed-forward regressor with residual connections, dropout, and batchnorm."""

import torch
from torch import nn

class EstimatorNN(nn.Module):
    """ResNet-like feed-forward network for regression."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 1, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        # Residual block
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        residual = out
        out = self.res_block(out)
        out = out + residual  # skip connection
        out = self.output_layer(out)
        return out

def EstimatorQNN() -> EstimatorNN:
    """Convenience factory returning the estimator network."""
    return EstimatorNN()

__all__ = ["EstimatorQNN"]
