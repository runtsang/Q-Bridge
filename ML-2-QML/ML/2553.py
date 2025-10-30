import torch
from torch import nn
import numpy as np
from.QuantumKernelMethod import Kernel as ClassicalKernel

class EstimatorNN(nn.Module):
    """Classical regressor that uses an RBF kernel to map inputs into a similarity space
    before a linear output.  The kernel is kept as a separate module to allow easy
    replacement with a quantum kernel in the future."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, gamma: float = 1.0):
        super().__init__()
        self.kernel = ClassicalKernel(gamma)
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        # Compute pairwise RBF similarity between batch `x` and a fixed support set.
        # `support` shape: (n_support, input_dim)
        diff = x.unsqueeze(1) - support.unsqueeze(0)          # (batch, n_support, dim)
        k = torch.exp(-self.kernel.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))
        # `k` shape: (batch, n_support, 1)
        return self.linear(k)

def EstimatorQNN() -> EstimatorNN:
    """Return an instance of the classical EstimatorNN."""
    return EstimatorNN()

__all__ = ["EstimatorNN", "EstimatorQNN"]
