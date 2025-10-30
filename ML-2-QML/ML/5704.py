import torch
from torch import nn
from typing import Iterable

class EstimatorNN(nn.Module):
    """
    A flexible feed‑forward regressor that accepts any number of input features
    and uses a residual‑like structure to capture non‑linear relationships.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_sizes: Iterable[int] = (8, 4),
                 output_dim: int = 1,
                 activation: nn.Module = nn.Tanh()):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(activation)
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def EstimatorQNN(input_dim: int = 2,
                 hidden_sizes: Iterable[int] = (8, 4),
                 output_dim: int = 1,
                 activation: nn.Module = nn.Tanh()) -> EstimatorNN:
    """
    Construct an EstimatorNN with the specified architecture.
    """
    return EstimatorNN(input_dim, hidden_sizes, output_dim, activation)

__all__ = ["EstimatorQNN", "EstimatorNN"]
