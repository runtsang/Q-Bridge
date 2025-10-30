"""Quantum kernel implementation using PennyLane and Qiskit Aer.

The module mirrors the original seed API while providing a fully
quantum‑only implementation that can be benchmarked against the
classical RBF baseline.  The key components are:

* :class:`PennyLaneKernel` – a QNode that computes the overlap
  ``|⟨ψ(x)|ψ(y)⟩|`` on a statevector simulator.
* :func:`kernel_matrix` – convenience wrapper that builds a Gram matrix.
* Compatibility aliases ``KernalAnsatz`` and ``Kernel`` that forward to
  :class:`PennyLaneKernel`.

The implementation uses the ``qiskit_aer`` backend to access a fast
statevector simulation, making it suitable for research experiments
on moderate‑size circuits.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence

# Import PennyLane with Qiskit Aer backend
import pennylane as qml
from pennylane import numpy as pnp

class PennyLaneKernel(nn.Module):
    """Quantum kernel based on a PennyLane circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (wires) in the circuit.
    depth : int
        Number of layers of rotations.
    device_name : str
        PennyLane device name, e.g. ``"qiskit.aer.statevector"``.
    """
    def __init__(self, num_qubits: int = 4, depth: int = 2,
                 device_name: str = "qiskit.aer.statevector") -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.dev = qml.device(device_name, wires=num_qubits)
        # QNode that returns the full statevector
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` and ``y`` into a quantum state."""
        # Encode x
        for i in range(self.num_qubits):
            qml.RY(x[i], wires=i)
        # Encode y with negative angles
        for i in range(self.num_qubits):
            qml.RY(-y[i], wires=i)
        return qml.state()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        state = self.qnode(x, y)
        # Return absolute amplitude of the |0...0⟩ component
        return torch.abs(state[0])

# Legacy aliases -------------------------------------------------------------
class KernalAnsatz(nn.Module):
    """Compatibility wrapper that forwards to :class:`PennyLaneKernel`."""
    def __init__(self, *_, **__):
        super().__init__()
        self.kernel = PennyLaneKernel()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, y)

class Kernel(nn.Module):
    """Compatibility wrapper that forwards to :class:`PennyLaneKernel`."""
    def __init__(self, *_, **__):
        super().__init__()
        self.kernel = PennyLaneKernel()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, y)

# Convenience function -------------------------------------------------------
def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor]) -> np.ndarray:
    """Build the Gram matrix using the PennyLane kernel."""
    kernel = PennyLaneKernel()
    return np.array([[kernel(x.unsqueeze(0), y.unsqueeze(0)).item() for y in b] for x in a])

__all__ = [
    "PennyLaneKernel",
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
]
