import numpy as np
import torch
from torch import nn
from typing import Iterable

class FCLGen299(nn.Module):
    """
    Extended fully connected layer with optional dropout and batch‑norm.
    Mimics the quantum layer interface via a `run` method while offering
    richer classical functionality.
    """

    def __init__(self, n_features: int = 1, dropout: float = 0.0, use_bn: bool = False):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.bn = nn.BatchNorm1d(1) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.dropout(x)
        x = self.bn(x)
        return torch.tanh(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Accept a list of parameters, feed them through the layer,
        and return a NumPy array, matching the quantum signature.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = self.forward(values)
        return out.detach().cpu().numpy().flatten()

__all__ = ["FCLGen299"]
