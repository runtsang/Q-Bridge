"""Hybrid RBF–Quantum kernel with learnable parameters.

The public API mirrors the original seed (`KernalAnsatz`, `Kernel`, `kernel_matrix`) but
adds a trainable gamma and a quantum‑kernel multiplier.  The two kernels are multiplied
element‑wise to produce a hybrid Gram matrix that can be optimised end‑to‑end with
PyTorch.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn, optim


# --------------------------------------------------------------------------- #
# Classical RBF component
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """Learnable RBF kernel with gamma as trainable weight."""

    def __init__(self, gamma: float = 1.0, learn_gamma: bool = False) -> None:
        super().__init__()
        self.gamma = nn.Parameter(
            torch.tensor(gamma, dtype=torch.float32), requires_grad=learn_gamma
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel value between two batches of shape (B, D)."""
        diff = x - y
        sq = torch.sum(diff**2, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * sq)

    def __repr__(self) -> str:
        return f"<KernalAnsatz gamma={self.gamma.item():.4f}>"


# --------------------------------------------------------------------------- #
# Hybrid kernel that multiplies classical RBF with a quantum kernel
# --------------------------------------------------------------------------- #
class Kernel(nn.Module):
    """
    Hybrid kernel that first evaluates the classical RBF and then multiplies it
    with a quantum kernel value.  The quantum kernel is passed as a callable
    that accepts two tensors and returns a scalar.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        learn_gamma: bool = False,
        quantum_kernel: callable | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.classical = KernalAnsatz(gamma, learn_gamma)
        self.quantum_kernel = quantum_kernel
        self.device = torch.device(device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return hybrid kernel value."""
        x = x.to(self.device)
        y = y.to(self.device)
        rbf = self.classical(x, y).squeeze()

        if self.quantum_kernel is None:
            return rbf

        # Expect quantum_kernel to return a scalar tensor
        q_val = self.quantum_kernel(x, y)
        return rbf * q_val

    def regularisation(self, lambda_reg: float = 0.0) -> torch.Tensor:
        """Optional L2 penalty on gamma."""
        if lambda_reg == 0.0:
            return torch.tensor(0.0, device=self.device)
        return lambda_reg * self.classical.gamma.pow(2)


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    gamma: float = 1.0,
    learn_gamma: bool = False,
    quantum_kernel: callable | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """
    Compute the Gram matrix between two lists of tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of data points (each element shape `(D,)`).
    gamma : float, optional
        Initial value for the RBF gamma.
    learn_gamma : bool, optional
        Whether to make gamma trainable.
    quantum_kernel : callable, optional
        Callable that implements a quantum kernel and returns a scalar tensor.
    device : str or torch.device, optional
        Device on which to perform the computation.
    """
    kernel = Kernel(gamma, learn_gamma, quantum_kernel, device)
    return np.array(
        [[kernel(a_i, b_j).item() for b_j in b] for a_i in a]
    )


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
