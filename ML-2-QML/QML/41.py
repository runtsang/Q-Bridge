"""Quantum kernel implementation using PennyLane.

The quantum kernel is a variational circuit that encodes two classical data vectors
(x and y) with opposite rotations and measures the probability of the all‑zero
state.  The returned value is a differentiable scalar that can be multiplied
with a classical RBF kernel in the hybrid model above.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pennylane as qml
import torch
from pennylane import numpy as pnp

# --------------------------------------------------------------------------- #
# Pennylane device and circuit
# --------------------------------------------------------------------------- #
n_wires = 4
dev = qml.device("default.qubit", wires=n_wires)


@qml.qnode(dev, interface="torch")
def _quantum_kernel_circuit(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Variational circuit that encodes x and y with opposite RY rotations."""
    for i in range(n_wires):
        qml.RY(x[i], wires=i)
    for i in range(n_wires):
        qml.RY(-y[i], wires=i)
    # Measure probability of all qubits in |0⟩
    return qml.probs(wires=range(n_wires))[0]


# --------------------------------------------------------------------------- #
# Quantum kernel wrapper
# --------------------------------------------------------------------------- #
class KernalAnsatz:
    """Quantum kernel that evaluates the above circuit."""

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of shape `(n_wires,)`.  They are automatically cast to
            float32 if necessary.
        """
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        return _quantum_kernel_circuit(x, y)


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix between two lists of tensors using the quantum kernel."""
    q_kernel = KernalAnsatz()
    return np.array(
        [[q_kernel(a_i, b_j).item() for b_j in b] for a_i in a]
    )


__all__ = ["KernalAnsatz", "kernel_matrix"]
