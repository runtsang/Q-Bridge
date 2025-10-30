"""Quantum kernel using Pennylane variational circuit.

The class implements a variational quantum kernel with learnable parameters.
It provides a `forward` method that computes the kernel value between two
feature vectors and a `kernel_matrix` function that builds the Gram matrix.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
import torch
from torch import nn
from typing import Sequence

class QuantumKernelMethod__gen110(nn.Module):
    """Variational quantum kernel with Pennylane."""
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, device: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device, wires=n_qubits)
        # Parameter shape: (n_layers, n_qubits, 3) for RX, RY, RZ
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3, dtype=torch.float32))
        # Create a qnode that uses the interface torch for autograd
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Quantum circuit that encodes two feature vectors and returns overlap."""
        # Encode x
        for i, wire in enumerate(range(self.n_qubits)):
            qml.RY(x[i], wires=wire)
        # Apply variational layers
        for layer in range(self.n_layers):
            for wire in range(self.n_qubits):
                qml.RX(self.params[layer, wire, 0], wires=wire)
                qml.RY(self.params[layer, wire, 1], wires=wire)
                qml.RZ(self.params[layer, wire, 2], wires=wire)
        # Encode y with negative angles
        for i, wire in enumerate(range(self.n_qubits)):
            qml.RY(-y[i], wires=wire)
        # Return overlap amplitude
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return kernel value between two feature vectors."""
        # The circuit returns a real number; we take absolute value to ensure positivity
        return torch.abs(self.qnode(x, y))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  n_qubits: int = 4, n_layers: int = 2) -> np.ndarray:
    """Compute Gram matrix using the variational quantum kernel."""
    qk = QuantumKernelMethod__gen110(n_qubits, n_layers)
    return np.array([[qk(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernelMethod__gen110", "kernel_matrix"]
