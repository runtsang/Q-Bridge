import pennylane as qml
import numpy as np

class QCNNModel:
    """Quantum QCNN implemented with Pennylane, featuring entangled convolution and pooling layers."""
    def __init__(self, n_qubits: int = 8, layers: int = 3, seed: int = 1234) -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        np.random.seed(seed)
        # Parameters: conv (layers, n_qubits//2, 3), pool (layers, n_qubits//4, 3)
        self.params = np.random.randn(layers * (n_qubits//2 * 3 + n_qubits//4 * 3))
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _conv_layer(self, wires, params):
        for i in range(0, len(wires), 2):
            idx = i // 2
            qml.RZ(params[idx, 0], wires=wires[i])
            qml.RY(params[idx, 1], wires=wires[i+1])
            qml.CNOT(wires=[wires[i], wires[i+1]])
            qml.RZ(params[idx, 2], wires=wires[i+1])

    def _pool_layer(self, source, sink, params):
        qml.CNOT(wires=[source, sink])
        qml.RZ(params[0], wires=source)
        qml.RY(params[1], wires=sink)
        qml.CNOT(wires=[sink, source])
        qml.RY(params[2], wires=sink)

    def _circuit(self, x, weights):
        # Feature map: rotate each qubit by input value
        for i, val in enumerate(x):
            qml.RY(val, wires=i)
        # Reshape weights
        conv_len = self.layers * (self.n_qubits // 2) * 3
        pool_len = self.layers * (self.n_qubits // 4) * 3
        conv_w = weights[:conv_len].reshape(self.layers, self.n_qubits // 2, 3)
        pool_w = weights[conv_len:conv_len+pool_len].reshape(self.layers, self.n_qubits // 4, 3)
        for l in range(self.layers):
            # Convolution
            for i in range(0, self.n_qubits // (2**l), 2):
                idx = i // 2
                qml.RZ(conv_w[l, idx, 0], wires=i)
                qml.RY(conv_w[l, idx, 1], wires=i+1)
                qml.CNOT(wires=[i, i+1])
                qml.RZ(conv_w[l, idx, 2], wires=i+1)
            # Pooling
            for i in range(0, self.n_qubits // (2**l), 2):
                idx = i // 2
                qml.CNOT(wires=[i, i+1])
                qml.RZ(pool_w[l, idx, 0], wires=i)
                qml.RY(pool_w[l, idx, 1], wires=i+1)
                qml.CNOT(wires=[i+1, i])
                qml.RY(pool_w[l, idx, 2], wires=i+1)
        return qml.expval(qml.PauliZ(0))

    def predict(self, x: np.ndarray) -> float:
        """Return the expectation value of Pauli‑Z on qubit 0."""
        return self.qnode(x, self.params)

    def train(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 200) -> None:
        """Simple gradient‑descent training loop for the QCNN."""
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for epoch in range(epochs):
            loss = 0.0
            for xi, yi in zip(X, y):
                pred = self.qnode(xi, self.params)
                loss += (pred - yi) ** 2
            loss /= len(X)
            self.params = opt.step(lambda p: self._loss(p, X, y), self.params)

    def _loss(self, params, X, y):
        loss = 0.0
        for xi, yi in zip(X, y):
            loss += (self.qnode(xi, params) - yi) ** 2
        return loss / len(X)

def QCNN() -> QCNNModel:
    """Factory returning a ready‑to‑train QCNNModel."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
