"""Quantum kernel implementation using PennyLane.

The module mirrors the structure of the original TorchQuantum example
but replaces the heavy TorchQuantum dependency with a lightweight
PennyLane QNode.  The kernel evaluates the overlap between two
state‑encoded data points and is used by the hybrid fraud detector
to augment its feature space.
"""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn

class KernalAnsatz(nn.Module):
    """Quantum data‑encoding ansatz using Ry rotations on 4 wires."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.dev = qml.device("default.qubit", wires=n_wires)
        # Build a simple QNode that encodes x on the first n_wires
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Encode positive data
            for i in range(self.n_wires):
                qml.RY(x[i], wires=i)
            # Encode negative data (inverse)
            for i in range(self.n_wires):
                qml.RY(-y[i], wires=i)
            return qml.probs(wires=range(self.n_wires))
        self.circuit = circuit

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the overlap (inner product) between the two encoded states."""
        probs = self.circuit(x, y)
        # For a pure state, the overlap is the square root of the
        # probability of measuring the all‑zero state.
        return torch.sqrt(probs[0])

class Kernel(nn.Module):
    """Convenience wrapper that exposes a `forward(x, y)` interface."""
    def __init__(self) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix between datasets `a` and `b`."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
