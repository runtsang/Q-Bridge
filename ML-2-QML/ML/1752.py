import torch
from torch import nn
import numpy as np
from typing import Iterable

class FCL(nn.Module):
    """
    An enhanced fully connected layer implemented in PyTorch.
    The network contains a configurable number of hidden units, dropout
    and a final linear layer that outputs a single scalar.
    """
    def __init__(self, n_features: int = 1, hidden_units: int = 32,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, 1)
        )

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Forward pass that accepts a list of parameters (thetas) and
        returns the mean of the network output.
        """
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            out = self.net(theta_tensor)
            expectation = torch.tanh(out).mean()
        return expectation.detach().numpy()

__all__ = ["FCL"]
