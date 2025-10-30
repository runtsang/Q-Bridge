from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RBFKernel(nn.Module):
    """Simple RBF kernel used in the classical hybrid sampler."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class HybridSamplerKernel(nn.Module):
    """
    Classical hybrid sampler that combines an MLP with an RBF kernel.
    The kernel modulates the output distribution of a small feed‑forward net.
    """
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.kernel = RBFKernel(gamma)
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        # Learnable kernel bias (analogous to weight_params in the quantum version)
        self.kernel_bias = nn.Parameter(torch.randn(2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Compute kernel similarity between the input and a learnable bias vector.
          2. Feed the raw input through the MLP.
          3. Modulate the MLP output by the kernel value.
          4. Return a probability distribution.
        """
        k = self.kernel(inputs, self.kernel_bias)
        out = self.net(inputs)
        out = out * k
        return F.softmax(out, dim=-1)

__all__ = ["HybridSamplerKernel"]
