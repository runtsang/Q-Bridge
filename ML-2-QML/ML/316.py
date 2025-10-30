import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Iterable

class FCL(nn.Module):
    """
    Extended fully‑connected module that supports arbitrary depth, dropout and
    batch‑normalisation.  It can be driven by an external *theta* vector to
    emulate a quantum‑style parameter sweep.
    """

    def __init__(self, input_dim: int = 1, hidden_dims: Iterable[int] | None = None,
                 dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        hidden_dims = hidden_dims or []
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h, bias=bias))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1, bias=bias))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Load the parameters from *thetas* and evaluate a single sample of shape
        (1, input_dim).  The vector is split according to the layer sizes.
        """
        # Flatten parameters
        params = torch.tensor(list(thetas), dtype=torch.float32).reshape(-1, 1)
        # Assign to model
        idx = 0
        for p in self.parameters():
            num = p.numel()
            p.data = params[idx:idx+num].reshape(p.shape)
            idx += num

        x = torch.ones(1, self.model[0].in_features)
        return self.forward(x).detach().numpy()

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

__all__ = ["FCL"]
