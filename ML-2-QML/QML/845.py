"""Quantum neural network using Pennylane with entanglement and variational layers.

The network operates on 2 qubits, encodes inputs with RX rotations,
applies a trainable variational layer, entangles the qubits,
and measures Pauli‑Z expectation values as outputs.
"""

import pennylane as qml
import numpy as np

# Device with 2 qubits
dev = qml.device("default.qubit", wires=2)

def _variational_circuit(inputs: np.ndarray, weights: np.ndarray) -> None:
    """Variational circuit with input encoding and entanglement."""
    # Input encoding
    qml.RX(inputs[0], wires=0)
    qml.RX(inputs[1], wires=1)

    # Variational rotations
    for i in range(len(weights)):
        qml.RY(weights[i], wires=i % 2)

    # Entanglement
    qml.CNOT(wires=[0, 1])

@qml.qnode(dev, interface="autograd")
def _qnode(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """QNode returning expectation values of Pauli‑Z on each qubit."""
    _variational_circuit(inputs, weights)
    return [qml.expval(qml.PauliZ(i)) for i in range(dev.num_wires)]

class EstimatorQNN:
    """Wrapper around the Pennylane QNode for regression tasks."""
    def __init__(self, weight_shape: tuple = (2,)) -> None:
        self.weight_shape = weight_shape
        self.weights = np.random.randn(*weight_shape)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the quantum circuit."""
        return _qnode(inputs, self.weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Batch prediction over an array of inputs."""
        return np.array([self.__call__(x) for x in X])

__all__ = ["EstimatorQNN"]
