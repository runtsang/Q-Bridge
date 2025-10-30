"""Hybrid classical sampler network using RBF kernel features and a softmax sampler.

This module defines a `HybridSamplerQNN` class that first transforms inputs via a
classical Radial Basis Function (RBF) kernel computed against a set of
training examples, then passes the resulting feature vector through a small
neural network that produces a probability distribution.  The design
combines the lightweight sampler from the original `SamplerQNN` with the
kernel machinery from `QuantumKernelMethod`, enabling a two‑stage
representation that can be trained end‑to‑end with standard PyTorch
optimizers.

The public API mirrors the original `SamplerQNN` helper so that existing
scripts can swap in the hybrid variant without modification:

```python
from SamplerQNN__gen083 import SamplerQNN
model = SamplerQNN()
```

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RBFKernel(nn.Module):
    """Classical RBF kernel with a learnable width parameter."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        # gamma is treated as a learnable parameter to allow the model
        # to adapt the kernel width during training.
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel matrix between two batches of vectors.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(m, d)``.
        y : torch.Tensor
            Shape ``(n, d)``.
        Returns
        -------
        torch.Tensor
            Shape ``(m, n)`` containing ``exp(-gamma * ||x-y||^2)``.
        """
        # Expand to compute pairwise squared distances efficiently
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (m, n, d)
        dist_sq = torch.sum(diff ** 2, dim=2)   # (m, n)
        return torch.exp(-self.gamma * dist_sq)

class SamplerNetwork(nn.Module):
    """Simple two‑layer softmax sampler."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over ``output_dim`` classes."""
        return F.softmax(self.net(z), dim=-1)

class HybridSamplerQNN(nn.Module):
    """
    Hybrid sampler that augments the classical sampler with a kernel‑based feature
    extractor.  The model first computes the similarity between a new input
    and a held‑out training set using an RBF kernel; the resulting feature
    vector is then passed to a small neural network that produces a
    probability distribution.

    Parameters
    ----------
    kernel : str, optional
        Type of kernel to use.  Currently only ``"rbf"`` is supported.
    gamma : float, optional
        Initial width of the RBF kernel.
    """
    def __init__(self, kernel: str = "rbf", gamma: float = 1.0):
        super().__init__()
        if kernel!= "rbf":
            raise ValueError(f"Unsupported kernel: {kernel!r}")
        self.kernel = RBFKernel(gamma)
        self.sampler = SamplerNetwork()
        # Buffers that hold the training data and corresponding output labels.
        self.register_buffer("train_X", torch.empty((0, 2)))
        self.register_buffer("train_y", torch.empty((0, 2)))

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Store training data in buffers.  ``X`` should be of shape ``(m, 2)``,
        ``y`` should be a one‑hot tensor of shape ``(m, 2)``.
        """
        self.train_X = X.clone().detach()
        self.train_y = y.clone().detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability distribution for a batch of inputs.

        The procedure is:

        1. If no training data has been provided, ``x`` is fed directly
           into the sampler.
        2. Otherwise, a kernel matrix ``K`` of shape ``(m, n)`` is computed
           between the stored training set ``self.train_X`` and the input
           batch ``x``.
        3. The kernel matrix is weighted by the stored one‑hot labels
           to produce a feature vector of shape ``(n, 2)``.
        4. The feature vector is passed through the sampler network.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(n, 2)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n, 2)`` – a probability distribution for each input.
        """
        if self.train_X.shape[0] == 0:
            # No training data: treat x as raw features.
            z = x
        else:
            K = self.kernel(self.train_X, x)          # (m, n)
            # Weighted sum of training labels.
            z = torch.matmul(self.train_y.t(), K)    # (2, n) → (n, 2)
            z = z.t()                                 # (n, 2)
        return self.sampler(z)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Expose the underlying RBF kernel for external use.

        Parameters
        ----------
        a, b : torch.Tensor
            Input tensors of shape ``(m, d)`` and ``(n, d)``.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape ``(m, n)``.
        """
        return self.kernel(a, b)

def SamplerQNN() -> HybridSamplerQNN:
    """
    Factory that returns a fresh instance of the hybrid sampler.

    The function signature matches the original ``SamplerQNN`` helper
    so that legacy code continues to work unchanged.
    """
    return HybridSamplerQNN()
