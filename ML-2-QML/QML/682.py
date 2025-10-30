"""Quantum kernel using PennyLane with a trainable variational circuit."""

import numpy as np
import torch
import pennylane as qml
from pennylane import numpy as npq
from typing import Sequence

class KernalAnsatz:
    """Variational ansatz for quantum kernel.

    The circuit encodes two input vectors x and y into rotation angles
    and applies a trainable entangling layer. The kernel value is
    the absolute overlap between the two resulting states.
    """
    def __init__(self, n_wires: int = 4, n_layers: int = 2):
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=self.n_wires)
        # Trainable parameters for entangling layer
        self.params = npq.random.randn(self.n_layers, self.n_wires, 3)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, y: torch.Tensor, params: torch.Tensor):
            # Encode x
            for i in range(self.n_wires):
                qml.RY(x[i], wires=i)
            # Entangling layer with parameters
            for l in range(self.n_layers):
                for i in range(self.n_wires):
                    qml.RZ(params[l, i, 0], wires=i)
                for i in range(self.n_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Encode -y
            for i in range(self.n_wires):
                qml.RY(-y[i], wires=i)
            return qml.state()

        self.circuit = circuit

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value for a single pair (x, y).

        Parameters
        ----------
        x : torch.Tensor
            Input vector of shape (n_wires,).
        y : torch.Tensor
            Input vector of shape (n_wires,).

        Returns
        -------
        torch.Tensor
            Absolute overlap between the two states.
        """
        state_x = self.circuit(x, torch.zeros_like(x), torch.tensor(self.params))
        state_y = self.circuit(torch.zeros_like(x), y, torch.tensor(self.params))
        overlap = torch.abs(torch.vdot(state_x, state_y))
        return overlap

class Kernel:
    """Quantum kernel wrapper."""
    def __init__(self, n_wires: int = 4, n_layers: int = 2):
        self.ansatz = KernalAnsatz(n_wires, n_layers)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], n_wires: int = 4, n_layers: int = 2) -> np.ndarray:
    """
    Compute Gram matrix for two collections of vectors using the quantum kernel.

    Parameters
    ----------
    a : Sequence[torch.Tensor]
        First collection of vectors.
    b : Sequence[torch.Tensor]
        Second collection of vectors.
    n_wires : int
        Number of qubits in the circuit.
    n_layers : int
        Depth of entangling layers.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    kernel = Kernel(n_wires, n_layers)
    A = torch.stack([x[:n_wires] for x in a])
    B = torch.stack([y[:n_wires] for y in b])
    mat = torch.zeros((len(a), len(b)), dtype=torch.float64)
    for i, xi in enumerate(A):
        for j, yj in enumerate(B):
            mat[i, j] = kernel(xi, yj).item()
    return mat.numpy()

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
