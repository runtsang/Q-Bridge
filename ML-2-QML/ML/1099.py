import torch
from torch import nn
import torch.nn.functional as F

class EstimatorQNN(nn.Module):
    """Enhanced feed‑forward regression network.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input feature vector.
    hidden_dims : list[int], default [64, 32]
        Sizes of successive hidden layers.
    dropout : float, default 0.0
        Dropout probability applied after each hidden layer.
    """
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | tuple[int] = (64, 32), dropout: float = 0.0) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

__all__ = ["EstimatorQNN"]
