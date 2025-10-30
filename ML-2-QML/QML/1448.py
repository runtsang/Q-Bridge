import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

# Global device for the small two‑qubit circuit
dev = qml.device("default.qubit", wires=2)

class Hybrid(nn.Module):
    """
    Quantum hybrid layer using a variational circuit.

    The layer evaluates the expectation value of Pauli‑Z on the first qubit
    after a simple entangling ansatz.  Pennylane's automatic differentiation
    (with interface='torch') provides gradients that can be back‑propagated
    through the quantum circuit.  The output is passed through a sigmoid
    to obtain a binary probability.
    """

    def __init__(self, shift: float = np.pi / 2, shots: int = 1000):
        super().__init__()
        self.shift = shift
        self.shots = shots
        # Define the variational circuit as a QNode
        self.qnode = qml.QNode(self._circuit, dev, interface="torch", diff_method="backprop")

    def _circuit(self, theta: torch.Tensor):
        # Entangling ansatz
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.CNOT(wires=[0, 1])
        # Parameterised rotation
        qml.RY(theta, wires=0)
        qml.RZ(theta, wires=1)
        # Return expectation value of Pauli‑Z on qubit 0
        return qml.expval(qml.PauliZ(wires=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure scalar input per sample
        x = x.squeeze()
        # Compute expectation value
        expval = self.qnode(x)
        probs = torch.sigmoid(expval)
        # Return 2‑class probability distribution
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["Hybrid"]
