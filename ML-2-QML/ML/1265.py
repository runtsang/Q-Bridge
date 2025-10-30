"""Classical implementation of a fully connected layer with gradient support."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable

class FullyConnectedLayer(nn.Module):
    """A fully connected layer with optional bias, dropout and gradient utilities."""
    def __init__(self, in_features: int, out_features: int = 1, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, self.linear.out_features)
        output = self.forward(theta_tensor)
        return output.detach().cpu().numpy()

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32, requires_grad=True).view(-1, self.linear.out_features)
        output = self.forward(theta_tensor)
        loss = output.mean()
        loss.backward()
        grad = theta_tensor.grad.detach().cpu().numpy()
        return grad

def FCL(in_features: int, out_features: int = 1, bias: bool = True, dropout: float = 0.0) -> FullyConnectedLayer:
    """Return a FullyConnectedLayer instance configured for the given dimensions."""
    return FullyConnectedLayer(in_features, out_features, bias=bias, dropout=dropout)

__all__ = ["FullyConnectedLayer", "FCL"]
