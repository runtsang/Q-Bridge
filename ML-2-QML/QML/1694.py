"""Quantum convolutional neural network implemented with Pennylane.

The circuit consists of parameterised rotation layers that act as
convolution steps, followed by entangling gates that realise pooling.
The design mirrors the classical stack but exploits quantum
parallelism and entanglement.
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class QCNNEnhanced:
    """Variational quantum circuit with convolution and pooling layers."""
    def __init__(self, num_qubits: int = 8, layers: int = 3, seed: int = 12345) -> None:
        self.num_qubits = num_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=num_qubits, shots=None)
        rng = np.random.default_rng(seed)
        # Parameters: layers × qubits × 3 rotation angles per qubit
        self.params = rng.uniform(0, 2 * np.pi, (layers, num_qubits, 3))
        self._build()

    def _build(self) -> None:
        @qml.qnode(self.dev, interface="autograd")
        def circuit(x):
            # Feature map: simple angle encoding
            for i, val in enumerate(x):
                qml.RY(val, wires=i)
            # Convolution + pooling stages
            for l in range(self.layers):
                self._conv_layer(range(self.num_qubits // (2 ** l)), l)
                self._pool_layer(range(self.num_qubits // (2 ** l)), l)
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def _conv_layer(self, wires, layer_idx):
        """Apply a convolutional block to the selected wires."""
        for i in range(0, len(wires), 2):
            w0, w1 = wires[i], wires[i + 1]
            qml.RZ(self.params[layer_idx][w0][0], w0)
            qml.RY(self.params[layer_idx][w0][1], w0)
            qml.CNOT(w0, w1)
            qml.RY(self.params[layer_idx][w1][2], w1)

    def _pool_layer(self, wires, layer_idx):
        """Apply a pooling block to the selected wires."""
        for i in range(0, len(wires) - 1, 2):
            w0, w1 = wires[i], wires[i + 1]
            qml.CNOT(w0, w1)
            qml.RZ(self.params[layer_idx][w0][0], w0)
            qml.RY(self.params[layer_idx][w1][1], w1)

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the QNN for a single data point."""
        return float(self.circuit(x))

def QCNNEnhanced() -> QCNNEnhanced:
    """Factory returning a configured :class:`QCNNEnhanced`."""
    return QCNNEnhanced()

__all__ = ["QCNNEnhanced", "QCNNEnhanced"]
