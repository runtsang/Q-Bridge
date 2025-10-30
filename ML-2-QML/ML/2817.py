"""Hybrid classical sampler with optional quantum kernel support.

This module builds on the original SamplerQNN by adding a kernel
module that can be either a classical RBF kernel or a quantum kernel
if the optional dependency is available.  The API mirrors the
original function ``SamplerQNN()`` but returns an instance of the
``HybridSamplerQNN`` class.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HybridSamplerQNN", "SamplerQNN"]

# Classical RBF kernel
class KernalAnsatz(nn.Module):
    """Gaussian RBF kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class HybridSamplerQNN(nn.Module):
    """Hybrid sampler that combines a small neural network with a kernel.

    Parameters
    ----------
    use_quantum_kernel : bool, optional
        If True and a quantum kernel implementation is importable,
        it will replace the classical RBF kernel.
    gamma : float, optional
        Width of the Gaussian kernel.  Used only when the classical
        kernel is active.
    """

    def __init__(self, use_quantum_kernel: bool = True, gamma: float = 1.0) -> None:
        super().__init__()
        # Mimic the original two‑layer sampler
        self.classical_net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

        # Decide on kernel type
        kernel = KernalAnsatz(gamma)
        if use_quantum_kernel:
            try:
                # Lazy import to avoid mandatory dependency on torchquantum
                from.quantum_module import Kernel as QuantumKernel
                kernel = QuantumKernel()
            except Exception:
                # Fall back to classical kernel
                pass
        self.kernel = kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute a probability distribution over two outcomes.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(batch, 2)``.
        y : torch.Tensor
            Reference batch used in the kernel evaluation,
            shape ``(batch, 2)``.

        Returns
        -------
        torch.Tensor
            Soft‑maxed kernel values of shape ``(batch, 2)``.
        """
        # Extract features with the small neural net
        feature = self.classical_net(x)
        # Evaluate kernel between features and reference
        k_val = self.kernel(feature, y)
        # Convert to probabilities
        return F.softmax(k_val, dim=-1)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the Gram matrix between two sets of samples."""
        return self.kernel(a, b)

def SamplerQNN() -> HybridSamplerQNN:
    """Factory compatible with the original API."""
    return HybridSamplerQNN()
