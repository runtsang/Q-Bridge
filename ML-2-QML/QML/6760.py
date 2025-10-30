"""Quantum convolution module.

Provides a Conv factory that returns a variational circuit
which encodes a 2‑D patch into a quantum state and
measures the expectation of Pauli‑Z on the first qubit.
"""

import numpy as np
import pennylane as qml

class _QuantumConv:
    """Variational circuit used for quanvolution layers."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.num_qubits = kernel_size ** 2
        # PennyLane device
        self.device = qml.device("default.qubit", wires=self.num_qubits)
        self.num_layers = 2
        # Random parameters for the variational ansatz
        self.params = np.random.uniform(0, 2 * np.pi, size=(self.num_layers, self.num_qubits))
        # Quantum node
        self.qnode = qml.QNode(self._circuit, self.device, interface="autograd")

    def _circuit(self, data: np.ndarray, params: np.ndarray) -> float:
        # Data encoding: RY rotations
        for i in range(self.num_qubits):
            qml.RY(data[i], wires=i)
        # Variational layers
        for layer in range(self.num_layers):
            for i in range(self.num_qubits):
                qml.RX(params[layer][i], wires=i)
            # Entanglement
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.num_qubits - 1, 0])
        # Expectation value of Pauli‑Z on the first qubit
        return qml.expval(qml.PauliZ(0))

    def run(self, data) -> float:
        """Run the quantum filter on a 2‑D array."""
        flat = np.asarray(data).flatten()
        # Encode data relative to the threshold
        encoded = np.where(flat > self.threshold, np.pi, 0.0)
        exp_val = self.qnode(encoded, self.params)
        return float(exp_val)

def Conv():
    """Factory that returns a QuantumConv instance."""
    return _QuantumConv()

__all__ = ["Conv"]
