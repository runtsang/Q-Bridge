"""Hybrid classical–quantum kernel method.

The class combines a learnable RBF kernel with an optional classical sampler
network.  It serves as a drop‑in replacement for the legacy
`QuantumKernelMethod` while exposing a richer interface that can be
plugged into downstream pipelines.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class RBFKernel(nn.Module):
    """Learnable radial basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # reshape to (n, d) for broadcasting
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist2 = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * dist2)


class SamplerQNN(nn.Module):
    """A lightweight classical sampler network used for data augmentation."""
    def __init__(self, in_features: int = 2, hidden: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)


class HybridKernelMethod(nn.Module):
    """Combines an RBF kernel with an optional sampler."""
    def __init__(self,
                 gamma: float = 1.0,
                 use_sampler: bool = False,
                 sampler_hidden: int = 4) -> None:
        super().__init__()
        self.kernel = RBFKernel(gamma)
        self.use_sampler = use_sampler
        if use_sampler:
            self.sampler = SamplerQNN(hidden=sampler_hidden)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Return Gram matrix between tensors a and b."""
        mat = self.kernel(a, b)
        return mat.detach().cpu().numpy()

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """Generate a probability distribution if a sampler is attached."""
        if not self.use_sampler:
            raise RuntimeError("Sampler not enabled in this instance.")
        return self.sampler(inputs)


__all__ = ["HybridKernelMethod", "RBFKernel", "SamplerQNN"]
