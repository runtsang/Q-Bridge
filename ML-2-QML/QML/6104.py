"""Quantum kernel using a parameterized variational circuit with Pennylane.

This implementation replaces the TorchQuantum ansatz with a Pennylane
QNode that supports automatic differentiation. The kernel is defined as
the absolute overlap of two encoded states. The class
`QuantumKernelMethod` is trainable and exposes a `forward` method that
returns a scalar kernel value, and a static `kernel_matrix` method to
compute the Gram matrix.
"""

import pennylane as qml
import numpy as np
import torch
from typing import Sequence

# Device with four qubits
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev, interface="torch")
def _kernel_qnode(x: torch.Tensor, y: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Variational circuit that computes the kernel."""
    # Encode data x
    for i in range(x.shape[0]):
        qml.RY(x[i], wires=i)
    # Apply variational rotations
    for i in range(params.shape[0]):
        qml.RZ(params[i], wires=i)
    # Entanglement layer
    for i in range(3):
        qml.CNOT(wires=[i, i + 1])
    # Encode -y
    for i in range(y.shape[0]):
        qml.RY(-y[i], wires=i)
    # Reverse variational rotations
    for i in range(params.shape[0]):
        qml.RZ(-params[i], wires=i)
    # Entanglement layer again
    for i in range(3):
        qml.CNOT(wires=[i, i + 1])
    # Return amplitude of |0000>
    return torch.abs(qml.state()[0])

class QuantumKernelMethod:
    """Quantum kernel wrapper using a Pennylane QNode."""
    def __init__(self, n_params: int = 4) -> None:
        self.params = torch.nn.Parameter(torch.randn(n_params))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _kernel_qnode(x, y, self.params)

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between collections of tensors."""
        kernel = QuantumKernelMethod()
        return np.array([[kernel.forward(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernelMethod"]
