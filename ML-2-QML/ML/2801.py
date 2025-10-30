"""Hybrid classical kernel combining RBF, sampler, and optional quantum guidance.

The module is intentionally lightweight: it imports the original
`Kernel` implementation from :mod:`QuantumKernelMethod` and augments it
with a trainable sampler network.  The sampler is a small feed‑forward
network that outputs a probability distribution; the dot product of
two such distributions is used as a multiplicative weight on the
classical RBF score.  The design keeps the kernel fully classical
(NumPy/Torch) while exposing a clear interface for future quantum
back‑ends.

The class is fully compatible with the original API:
    from HybridKernelMethod import HybridKernel
    h = HybridKernel(gamma=0.5, use_sampler=True)
    h.kernel_matrix(X, Y)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Import the original RBF kernel implementation
try:
    from QuantumKernelMethod import Kernel as ClassicalKernel
except Exception:  # pragma: no cover
    # Fallback: define a minimal RBF kernel if the original module is missing.
    class ClassicalKernel(nn.Module):
        def __init__(self, gamma: float = 1.0) -> None:
            super().__init__()
            self.gamma = gamma

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def SamplerQNN() -> nn.Module:
    """Small softmax classifier used as a tunable kernel weight."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            # Softmax over the last dimension to produce a probability vector.
            return F.softmax(self.net(inputs), dim=-1)

    return SamplerModule()

class HybridKernel(nn.Module):
    """Hybrid kernel that optionally multiplies RBF similarity by a sampler score.

    Parameters
    ----------
    gamma : float
        RBF kernel bandwidth.
    use_sampler : bool
        If ``True``, a trainable sampler network is applied to each input
        before computing the dot‑product weight.
    """

    def __init__(self, gamma: float = 1.0, use_sampler: bool = False) -> None:
        super().__init__()
        self.rbf = ClassicalKernel(gamma)
        self.use_sampler = use_sampler
        if self.use_sampler:
            self.sampler = SamplerQNN()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return a scalar kernel value for two input vectors.

        The inputs are expected to be 1‑D tensors of equal length.
        """
        # RBF similarity
        rbf_score = self.rbf(x, y)

        if self.use_sampler:
            # Compute sampler outputs (probability distributions)
            s_x = self.sampler(x)
            s_y = self.sampler(y)
            # Dot product of probability vectors
            sampler_score = torch.sum(s_x * s_y, dim=-1, keepdim=True)
            return rbf_score * sampler_score
        return rbf_score

    def kernel_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the Gram matrix between two sets of samples.

        Parameters
        ----------
        a, b : array‑like
            Input data arrays of shape (n_samples, n_features).
        """
        a_t = torch.tensor(a, dtype=torch.float32)
        b_t = torch.tensor(b, dtype=torch.float32)
        return np.array([[self.forward(x, y).item() for y in b_t] for x in a_t])

__all__ = ["HybridKernel"]
