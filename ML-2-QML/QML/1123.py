import pennylane as qml
import numpy as np
import torch
from typing import Sequence

class QuantumKernelMethod:
    """Quantum kernel using a variational circuit.

    The ansatz consists of layers of Ry rotations followed by CNOT
    entanglement.  All parameters are trainable and can be optimized
    jointly with a downstream learning algorithm.  The kernel value
    is the absolute overlap of the two encoded states.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2,
                 device_name: str = 'default.qubit') -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device_name, wires=n_qubits)
        # trainable parameters for the variational circuit
        self.params = qml.numpy.array(
            np.random.randn(n_layers, n_qubits, 1),
            requires_grad=True
        )

        @qml.qnode(self.dev, interface='torch')
        def circuit(x):
            # data encoding
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)
            # variational layers
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RY(self.params[layer, qubit, 0], wires=qubit)
                # entanglement
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            return qml.state()

        self.circuit = circuit

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value ``k(x, y) = |<ψ(x)|ψ(y)>|``."""
        x = x.squeeze()
        y = y.squeeze()
        state_x = self.circuit(x)
        state_y = self.circuit(y)
        overlap = torch.dot(state_x.conj(), state_y)
        return torch.abs(overlap)

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor],
                      n_qubits: int = 4,
                      n_layers: int = 2,
                      device_name: str = 'default.qubit') -> np.ndarray:
        """Compute the Gram matrix between two collections of vectors."""
        qk = QuantumKernelMethod(n_qubits, n_layers, device_name)
        a = [torch.tensor(v, dtype=torch.float32) for v in a]
        b = [torch.tensor(v, dtype=torch.float32) for v in b]
        K = np.zeros((len(a), len(b)))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                K[i, j] = qk(x, y).item()
        return K

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
