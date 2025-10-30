import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence

class RBFKernel(nn.Module):
    """Classical radial basis function kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class SamplerQNNHybrid(nn.Module):
    """Hybrid sampler combining a classical network and an RBF kernel."""
    def __init__(self, hidden_dim: int = 4, gamma: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.kernel = RBFKernel(gamma)

    def forward(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(self.net(inputs), dim=-1)
        similarity = self.kernel(inputs, weights)
        return probs * similarity

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix using the RBF kernel."""
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["SamplerQNNHybrid", "kernel_matrix"]
