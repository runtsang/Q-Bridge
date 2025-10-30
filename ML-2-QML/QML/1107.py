import pennylane as qml
import pennylane.numpy as np
from pennylane import qnode

class QCNNHybrid:
    """Quantum neural network with feature‑map and variational ansatz inspired by QCNN."""
    def __init__(self, num_qubits: int = 8, layers: int = 3, shots: int = 1000):
        self.num_qubits = num_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=num_qubits, shots=shots)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Feature map
            for i, w in enumerate(inputs):
                qml.RY(w, wires=i)
            # Ansatz layers
            for l in range(self.layers):
                for i in range(self.num_qubits):
                    qml.RZ(weights[l, i], wires=i)
                for i in range(0, self.num_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        self.weights = np.random.randn(layers, num_qubits)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.circuit(inputs, self.weights)

def QCNN() -> QCNNHybrid:
    """Factory returning the configured :class:`QCNNHybrid`."""
    return QCNNHybrid()

__all__ = ["QCNN", "QCNNHybrid"]
