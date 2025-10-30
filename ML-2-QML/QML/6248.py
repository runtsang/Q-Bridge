"""Quantum kernel using PennyLane.

The class implements a simple overlap‑based kernel that encodes
classical data via a rotation‑only ansatz.  The overlap
between the states produced by two data points is returned as the
kernel value.  The implementation is fully differentiable and can
be trained end‑to‑end with PyTorch.
"""

from __future__ import annotations

import numpy as np
import torch
import pennylane as qml
from typing import Sequence

class QuantumKernelMethod:
    """Overlap kernel implemented with PennyLane.

    The kernel is defined as the probability of measuring the all‑zero
    computational basis state after applying the encoding for ``x``
    followed by the inverse encoding for ``y``.  This is equivalent to
    the squared absolute value of the overlap ⟨ψ(x)|ψ(y)⟩.
    """

    def __init__(self, n_wires: int = 4) -> None:
        self.n_wires = n_wires
        self.dev = qml.device("default.qubit", wires=self.n_wires)

        @qml.qnode(self.dev, interface="torch")
        def _kernel_qnode(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Encode x
            for i, val in enumerate(x):
                qml.RY(val, wires=i)
            # Encode -y (inverse encoding)
            for i, val in enumerate(y):
                qml.RY(-val, wires=i)
            # Probability of |0...0>
            probs = qml.probs(wires=range(self.n_wires))
            return probs[0]

        self._kernel_qnode = _kernel_qnode

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the kernel value for two 1‑D tensors."""
        x = x.view(-1)
        y = y.view(-1)
        return self._kernel_qnode(x, y)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  n_wires: int = 4) -> np.ndarray:
    """Compute the Gram matrix between two data sets.

    Parameters
    ----------
    a, b : sequences of 1‑D torch tensors
    n_wires : int
        Number of qubits used by the kernel.  The length of the input
        tensors must not exceed this number.
    """
    kernel = QuantumKernelMethod(n_wires=n_wires)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
