"""QCNN: Pennylane implementation of a quantum convolution‑pooling network.

The class builds a variational circuit that closely follows the classical depth and structure: a feature map, three convolutional layers, three pooling layers, and a final measurement on the first qubit.  It exposes a `predict` method that returns the expectation value of `Z` on wire 0, and supports parameter updates via `set_params`.  The implementation uses Pennylane's autograd interface for seamless integration with gradient‑based optimisers.
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class QCNN:
    """Quantum convolution‑pooling network implemented with Pennylane."""
    def __init__(self, n_qubits: int = 8, seed: int = 12345) -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=None)
        # Initialise weight parameters for all convolution and pooling blocks
        self.params = pnp.random.uniform(0, 2 * np.pi, size=(self.n_qubits * 3,))
        self.feature_map = qml.templates.embeddings.ZFeatureMap(n_qubits, reps=1)
        self._build_circuit()

    def _conv_block(self, wires: list[int], params: np.ndarray) -> None:
        """Convolution block acting on two wires."""
        qml.RZ(-np.pi / 2, wires[1])
        qml.CNOT(wires[1], wires[0])
        qml.RZ(params[0], wires[0])
        qml.RY(params[1], wires[1])
        qml.CNOT(wires[0], wires[1])
        qml.RY(params[2], wires[1])
        qml.CNOT(wires[1], wires[0])
        qml.RZ(np.pi / 2, wires[0])

    def _pool_block(self, wires: list[int], params: np.ndarray) -> None:
        """Pooling block acting on two wires."""
        qml.RZ(-np.pi / 2, wires[1])
        qml.CNOT(wires[1], wires[0])
        qml.RZ(params[0], wires[0])
        qml.RY(params[1], wires[1])
        qml.CNOT(wires[0], wires[1])
        qml.RY(params[2], wires[1])

    def _build_circuit(self) -> None:
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> float:
            # Feature map
            self.feature_map(inputs)
            idx = 0
            # Layer 1: 4 qubits → 4 conv blocks
            for i in range(0, self.n_qubits, 2):
                self._conv_block([i, i + 1], weights[idx : idx + 3])
                idx += 3
            # Pooling layer 1
            for i in range(0, self.n_qubits, 2):
                self._pool_block([i, i + 1], weights[idx : idx + 3])
                idx += 3
            # Layer 2: 4 qubits → 2 conv blocks
            for i in range(self.n_qubits // 2, self.n_qubits, 2):
                self._conv_block([i, i + 1], weights[idx : idx + 3])
                idx += 3
            # Pooling layer 2
            for i in range(self.n_qubits // 2, self.n_qubits, 2):
                self._pool_block([i, i + 1], weights[idx : idx + 3])
                idx += 3
            # Layer 3: 2 qubits → 1 conv block
            self._conv_block([self.n_qubits - 2, self.n_qubits - 1], weights[idx : idx + 3])
            idx += 3
            # Pooling layer 3
            self._pool_block([self.n_qubits - 2, self.n_qubits - 1], weights[idx : idx + 3])
            # Measurement
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def predict(self, inputs: np.ndarray, weights: np.ndarray | None = None) -> float:
        """Return the expectation value of Z on wire 0 for given inputs."""
        if weights is None:
            weights = self.params
        return self.circuit(inputs, weights)

    def set_params(self, params: np.ndarray) -> None:
        """Update the circuit weights."""
        self.params = params

def QCNN() -> QCNN:
    """Factory returning a ready‑to‑train quantum QCNN instance."""
    return QCNN()

__all__ = ["QCNN"]
