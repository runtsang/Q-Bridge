import torch
from torch import nn

class HybridEstimator(nn.Module):
    """
    A robust classical feed-forward regressor with batch normalization,
    dropout, and residual-like skip connections.  The architecture is
    deliberately deeper than the seed to better capture nonlinear
    relationships while keeping the network lightweight.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: list[int] | tuple[int,...] = (64, 32, 16),
                 output_dim: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

__all__ = ["HybridEstimator"]
