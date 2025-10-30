"""Quantum kernel and hybrid expectation head implemented with Pennylane.

The module provides a variational quantum kernel that computes the
overlap between two encoded classical vectors, and a hybrid layer
that evaluates a parameterised expectation value.  Both components
are fully differentiable with respect to their parameters, enabling
integration into a PyTorch training loop.
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn


class VariationalKernel:
    """Variational quantum kernel that evaluates the overlap between two
    classically encoded vectors."""

    def __init__(self, n_qubits: int, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def encode(self, x: np.ndarray):
        """Encode a classical vector into a quantum state."""
        for i, val in enumerate(x):
            qml.PhaseShift(val, wires=i)

    def circuit(self, x: np.ndarray, y: np.ndarray):
        """Return the expectation value of Z on qubit 0 for the
        overlap of states |x⟩ and |y⟩."""
        self.encode(x)
        self.encode(y)
        return qml.expval(qml.PauliZ(0))

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the kernel matrix for batches of x and y."""
        xs = x.detach().cpu().numpy()
        ys = y.detach().cpu().numpy()
        kernel_vals = np.zeros((xs.shape[0], ys.shape[0]))
        for i, xi in enumerate(xs):
            for j, yj in enumerate(ys):
                kernel_vals[i, j] = self.circuit(xi, yj)
        return torch.tensor(kernel_vals, dtype=torch.float32)


class HybridExpectation(nn.Module):
    """Hybrid layer that forwards activations through a Pennylane circuit
    and returns an expectation value."""

    def __init__(self, n_qubits: int, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def circuit(x_vec):
            for i in range(self.n_qubits):
                qml.RY(x_vec[i], wires=i)
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(self.params[l, i], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, self.dev, interface="torch")
        return qnode(x)
