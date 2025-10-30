import torch
import torch.nn as nn
import numpy as np

class RBFKernel(nn.Module):
    """Classical RBF kernel for distance‑based weighting."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class SamplerQNN(nn.Module):
    """Hybrid classical sampler that maps inputs to quantum circuit parameters."""
    def __init__(self, n_features: int = 2, hidden_dim: int = 8, n_params: int = 4, gamma: float = 1.0):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_params),
        )
        self.kernel = RBFKernel(gamma)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Map input features to a vector of quantum parameters.
        """
        return self.feature_extractor(inputs)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """
        Compute the Gram matrix between two sets of samples.
        """
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])
