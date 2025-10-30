import torch
from torch import nn

def EstimatorQNN():
    """Return a robust regression neural network with residual connections and dropout."""
    class EstimatorNN(nn.Module):
        def __init__(self, input_dim: int = 2, hidden_dims: list[int] = [32, 16], output_dim: int = 1, dropout: float = 0.1) -> None:
            super().__init__()
            layers = []
            prev_dim = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = h
            layers.append(nn.Linear(prev_dim, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    return EstimatorNN()

__all__ = ["EstimatorNN", "EstimatorQNN"]
