import numpy as np
import torch
from torch import nn
from typing import Iterable

class FullyConnectedLayer(nn.Module):
    """
    Two‑layer fully connected network with optional dropout.
    """
    def __init__(self, n_features: int = 1, n_hidden: int = 32, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_hidden, 1)
        )

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Forward pass with a list of parameters.
        The parameters are reshaped into a tensor and fed through the network.
        Returns the mean activation as a NumPy array.
        """
        x = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = self.net(x)
        expectation = out.mean(dim=0)
        return expectation.detach().numpy()

def FCL():
    """
    Factory that returns an instance of the fully connected layer.
    """
    return FullyConnectedLayer()

__all__ = ["FullyConnectedLayer"]
