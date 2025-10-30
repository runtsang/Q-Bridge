import torch
from torch import nn
import numpy as np
from typing import Iterable

class FCL(nn.Module):
    """
    Classical fully connected network with optional dropout.
    Configurable depth and hidden sizes.
    """
    def __init__(self, input_dim: int = 1, hidden_dims: Iterable[int] = (32, 16), output_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Accepts a list of parameters, feeds them as a single‑column tensor
        through the network and returns the mean output as a NumPy array.
        """
        theta_tensor = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            out = self.forward(theta_tensor)
        return out.mean(dim=0).cpu().numpy()

__all__ = ["FCL"]
