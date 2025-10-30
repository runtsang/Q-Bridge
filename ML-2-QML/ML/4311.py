"""Hybrid classical sampler with kernel and fully connected layer support.

This module merges ideas from the original SamplerQNN, a classical RBF kernel,
and a fully connected layer.  The `HybridSamplerQNN` class can:
* sample from a softmax distribution produced by a small feed‑forward network,
* evaluate a Gram matrix via a classical RBF kernel, and
* compute expectations through a lightweight fully‑connected layer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Sequence


class RBFKernel(nn.Module):
    """Classical RBF kernel used for Gram matrix construction."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class FullyConnectedLayer(nn.Module):
    """Simple fully‑connected layer that returns a scalar expectation."""

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()


class HybridSamplerQNN(nn.Module):
    """
    Hybrid sampler that combines a classical neural network, an RBF kernel,
    and a fully‑connected layer.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    hidden_dim : int
        Size of the hidden layer in the sampler network.
    kernel_gamma : float
        Hyper‑parameter for the RBF kernel.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 4,
        kernel_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.sampler_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.kernel = RBFKernel(kernel_gamma)
        self.fcl = FullyConnectedLayer(n_features=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over the input dimension."""
        logits = self.sampler_net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Draw samples from the softmax distribution."""
        probs = self.forward(inputs)
        return torch.multinomial(probs, num_samples, replacement=True)

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Compute the Gram matrix between two batches of samples."""
        a = torch.stack(a)
        b = torch.stack(b)
        diff = a.unsqueeze(1) - b.unsqueeze(0)  # shape (len(a), len(b), dim)
        k = torch.exp(-self.kernel.gamma * torch.sum(diff * diff, dim=-1))
        return k.detach().cpu().numpy()

    def run_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        """Delegate to the fully‑connected layer."""
        return self.fcl.run(thetas)


def SamplerQNN() -> HybridSamplerQNN:
    """Return a ready‑to‑use instance of the hybrid sampler."""
    return HybridSamplerQNN()


__all__ = ["HybridSamplerQNN", "SamplerQNN", "RBFKernel", "FullyConnectedLayer"]
