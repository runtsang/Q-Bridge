"""Quantum neural network with a 4‑qubit entangled ansatz and Pauli‑Z measurement."""
import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class EstimatorQNNGen409:
    """Quantum neural network with angle encoding, entangling layers and a single‑qubit observable."""
    def __init__(self, dev: qml.Device | None = None, num_qubits: int = 4):
        if dev is None:
            dev = qml.device("default.qubit", wires=num_qubits)
        self.dev = dev
        self.num_qubits = num_qubits
        self.qnode = qml.QNode(self._circuit, dev)

    def _circuit(self, x: list[float], weights: np.ndarray):
        # Angle‑encoding of classical features
        for i, wire in enumerate(range(self.num_qubits)):
            qml.RX(x[i % len(x)], wires=wire)
        # First entangling layer
        for i in range(self.num_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.num_qubits])
        # Variational parameters
        for i in range(self.num_qubits):
            qml.RY(weights[i], wires=i)
        # Second entangling layer
        for i in range(self.num_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.num_qubits])
        # Expectation value of Pauli‑Z on wire 0
        return qml.expval(qml.PauliZ(0))

    def __call__(self, x: list[float], weights: np.ndarray) -> float:
        """Evaluate the circuit and return the expectation value."""
        return self.qnode(x, weights)

def EstimatorQNNGen409() -> EstimatorQNNGen409:
    """Convenience factory mirroring the original API."""
    return EstimatorQNNGen409()

__all__ = ["EstimatorQNNGen409"]
