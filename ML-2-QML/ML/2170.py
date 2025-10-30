import torch
from torch import nn

class EstimatorQNNGen391(nn.Module):
    """Deep regression network with dropout and configurable hidden layers."""
    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int,...] = (64, 32, 16), dropout: float = 0.2) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def EstimatorQNN() -> EstimatorQNNGen391:
    """Factory that returns an instance of the extended estimator."""
    return EstimatorQNNGen391()

__all__ = ["EstimatorQNNGen391", "EstimatorQNN"]
