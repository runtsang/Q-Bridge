import pennylane as qml
import pennylane.numpy as np
from pennylane.optimize import AdamOptimizer

def QCNN(num_qubits: int = 8, seed: int = 12345):
    """
    Create a PennyLane QNode implementing the QCNN architecture.
    The returned QNode takes an input vector of length 8 and a flat
    array of 42 trainable parameters and returns a scalar in [−1,1].
    """
    dev = qml.device("default.qubit", wires=num_qubits)
    np.random.seed(seed)

    def conv_layer(start_wire: int, num_pairs: int, weights: np.ndarray, offset: int):
        for i in range(num_pairs):
            w = weights[offset + 3 * i: offset + 3 * i + 3]
            qml.RZ(-np.pi / 2, start_wire + 2 * i + 1)
            qml.CNOT(start_wire + 2 * i + 1, start_wire + 2 * i)
            qml.RZ(w[0], start_wire + 2 * i)
            qml.RY(w[1], start_wire + 2 * i + 1)
            qml.CNOT(start_wire + 2 * i, start_wire + 2 * i + 1)
            qml.RY(w[2], start_wire + 2 * i + 1)
            qml.CNOT(start_wire + 2 * i + 1, start_wire + 2 * i)
            qml.RZ(np.pi / 2, start_wire + 2 * i)

    def pool_layer(start_wire: int, num_pairs: int, weights: np.ndarray, offset: int):
        for i in range(num_pairs):
            w = weights[offset + 3 * i: offset + 3 * i + 3]
            qml.RZ(-np.pi / 2, start_wire + 2 * i + 1)
            qml.CNOT(start_wire + 2 * i + 1, start_wire + 2 * i)
            qml.RZ(w[0], start_wire + 2 * i)
            qml.RY(w[1], start_wire + 2 * i + 1)
            qml.CNOT(start_wire + 2 * i, start_wire + 2 * i + 1)
            qml.RY(w[2], start_wire + 2 * i + 1)

    @qml.qnode(dev)
    def circuit(x: np.ndarray, weights: np.ndarray) -> float:
        # Feature map: simple Z‑feature map
        for i, xi in enumerate(x):
            qml.Hadamard(i)
            qml.RZ(xi, i)

        # Convolution‑layer 1 (8 qubits)
        conv_layer(0, 4, weights, 0)
        # Pooling‑layer 1 (8 → 4)
        pool_layer(0, 4, weights, 12)

        # Convolution‑layer 2 (4 qubits, indices 4‑7)
        conv_layer(4, 2, weights, 24)
        # Pooling‑layer 2 (4 → 2)
        pool_layer(4, 2, weights, 30)

        # Convolution‑layer 3 (2 qubits, indices 6‑7)
        conv_layer(6, 1, weights, 36)
        # Pooling‑layer 3 (2 → 1)
        pool_layer(6, 1, weights, 39)

        # Return expectation value of Pauli‑Z on wire 0
        return qml.expval(qml.PauliZ(0))

    return circuit

class QCNNEnhanced:
    """
    Convenience wrapper around the QCNN QNode, mirroring the classical
    QCNNEnhanced class.  It allows the quantum circuit to be called like
    a normal PyTorch module.
    """
    def __init__(self, num_qubits: int = 8, seed: int = 12345):
        self.qnode = QCNN(num_qubits, seed)

    def __call__(self, x: np.ndarray, weights: np.ndarray) -> float:
        return self.qnode(x, weights)

    @staticmethod
    def create_qnode(num_qubits: int = 8, seed: int = 12345):
        return QCNN(num_qubits, seed)

__all__ = ["QCNN", "QCNNEnhanced"]
