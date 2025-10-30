"""Enhanced fully connected layer with dropout and batch normalization."""
import numpy as np
import torch
from torch import nn
from typing import Iterable, Optional

class FCLayer(nn.Module):
    """
    Classical fully connected layer that accepts a list of parameters
    and returns the mean of a tanh activation.
    Supports optional dropout and batch normalization to improve
    generalisation and stability.
    """
    def __init__(self, n_features: int = 1, dropout: float = 0.0, batch_norm: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.batch_norm = nn.BatchNorm1d(1) if batch_norm else nn.Identity()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the layer with the provided parameters."""
        values = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        x = self.linear(values)
        x = self.dropout(x)
        x = self.batch_norm(x)
        expectation = torch.tanh(x).mean(dim=0)
        return expectation.detach().cpu().numpy()

def FCL(n_features: int = 1, dropout: float = 0.0, batch_norm: bool = False) -> FCLayer:
    """Factory that returns an instance of the enhanced fully connected layer."""
    return FCLayer(n_features, dropout, batch_norm)

__all__ = ["FCL"]
