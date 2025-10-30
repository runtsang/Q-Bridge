import torch
from torch import nn
import numpy as np
from typing import Iterable, Callable, List

class FullyConnectedLayer(nn.Module):
    """
    Enhanced fully connected layer mimicking a quantum layer.
    Supports optional dropout, custom activation, and batch inference.
    """
    def __init__(self,
                 n_features: int = 1,
                 dropout: float = 0.0,
                 activation: Callable = torch.tanh,
                 device: str = "cpu") -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1).to(device)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.activation = activation
        self.device = device

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Forward pass for a single parameter vector.
        """
        theta_tensor = torch.as_tensor(list(thetas),
                                      dtype=torch.float32,
                                      device=self.device).view(-1, 1)
        x = self.linear(theta_tensor)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.activation(x).mean(dim=0)

    def run(self, thetas: Iterable[Iterable[float]]) -> np.ndarray:
        """
        Accepts a batch of parameter vectors and returns a NumPy array of
        expectation values.
        """
        outputs = [self.forward(t).item() for t in thetas]
        return np.array(outputs)

__all__ = ["FullyConnectedLayer"]
