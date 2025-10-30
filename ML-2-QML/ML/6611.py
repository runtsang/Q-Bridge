"""Enhanced classical and hybrid kernel module."""

from __future__ import annotations

import torch
import numpy as np
from torch import nn
from typing import Sequence, Callable, Optional

class QuantumKernelMethod(nn.Module):
    """
    Classical RBF kernel with optional hybrid blending with a quantum kernel.

    Parameters
    ----------
    gamma : float, optional
        Initial value for the RBF gamma parameter. If None, gamma will be a learnable parameter.
    weight : float, optional
        Weight for the quantum kernel in hybrid mode. If None, weight is a learnable parameter.
    mode : str, optional
        One of 'classical', 'quantum', 'hybrid'. In 'quantum' mode the class delegates to a supplied
        qml_kernel callable. In 'hybrid' mode it blends the classical and quantum kernels.
    qml_kernel : Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
        Callable that computes the quantum kernel between two tensors. Required if mode is 'quantum'
        or 'hybrid'.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        weight: float = 0.5,
        mode: str = "classical",
        qml_kernel: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        # gamma can be trainable
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32)) if isinstance(gamma, (int, float)) else gamma
        # weight for hybrid blending
        self.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32)) if isinstance(weight, (int, float)) else weight
        self.qml_kernel = qml_kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value between two input tensors.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of shape (n_features,).

        Returns
        -------
        torch.Tensor
            Kernel value as a scalar tensor.
        """
        if self.mode == "classical":
            return self._rbf_single(x, y)
        elif self.mode == "quantum":
            if self.qml_kernel is None:
                raise ValueError("qml_kernel callable must be provided in quantum mode.")
            return self.qml_kernel(x, y)
        elif self.mode == "hybrid":
            if self.qml_kernel is None:
                raise ValueError("qml_kernel callable must be provided in hybrid mode.")
            return self.weight * self._rbf_single(x, y) + (1.0 - self.weight) * self.qml_kernel(x, y)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def _rbf_single(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff)).squeeze()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two sequences of tensors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of 1‑D tensors where each tensor is of shape (n_features,).

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        matrix = torch.tensor([[self.forward(x, y) for y in b] for x in a])
        return matrix.detach().cpu().numpy()

__all__ = ["QuantumKernelMethod"]
