import pennylane as qml
import pennylane.numpy as np
from pennylane import numpy as pnp

class QCNN:
    """Quantum convolutional neural network using Pennylane."""
    def __init__(self, num_qubits: int = 8, layers: int = 3, shots: int = 1024):
        self.num_qubits = num_qubits
        self.layers = layers
        self.device = qml.device("default.qubit", wires=num_qubits, shots=shots)
        self.params = np.random.uniform(0, 2 * np.pi, size=(layers, num_qubits, 3))
        self.feature_map = qml.templates.embeddings.ZFeatureMap(num_qubits)
        self._build_circuit()

    def _conv_layer(self, wires, params):
        for i in range(0, len(wires) - 1, 2):
            qml.RZ(params[i, 0], wires[i])
            qml.CNOT(wires[i], wires[i + 1])
            qml.RY(params[i, 1], wires[i + 1])
            qml.CNOT(wires[i + 1], wires[i])
            qml.RY(params[i, 2], wires[i + 1])

    def _pool_layer(self, wires, params):
        for i in range(0, len(wires) - 1, 2):
            qml.RZ(params[i, 0], wires[i])
            qml.CNOT(wires[i], wires[i + 1])
            qml.RY(params[i, 1], wires[i + 1])
            qml.CNOT(wires[i + 1], wires[i])
            qml.RY(params[i, 2], wires[i + 1])

    def _build_circuit(self):
        @qml.qnode(self.device, interface="autograd")
        def circuit(x, weights):
            self.feature_map(x)
            for l in range(self.layers):
                self._conv_layer(range(self.num_qubits), weights[l])
                if l < self.layers - 1:
                    for w in range(1, self.num_qubits, 2):
                        qml.expval(qml.PauliZ(w))
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def predict(self, x: np.ndarray) -> float:
        probs = self.circuit(x, self.params)
        return float((probs + 1) / 2)

    def train(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 200):
        opt = qml.GradientDescentOptimizer(lr)
        for _ in range(epochs):
            def cost(weights):
                loss = 0.0
                for xi, yi in zip(X, y):
                    pred = self.circuit(xi, weights)
                    loss += (pred - yi) ** 2
                return loss / len(X)
            self.params = opt.step(cost, self.params)

__all__ = ["QCNN"]
