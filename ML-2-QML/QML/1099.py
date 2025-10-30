import pennylane as qml
import numpy as np

class EstimatorQNN:
    """Quantum variational regressor.

    Features
    --------
    num_qubits : int, default 2
        Number of qubits in the circuit.
    layers : int, default 2
        Number of variational layers.
    """
    def __init__(self, num_qubits: int = 2, layers: int = 2) -> None:
        self.num_qubits = num_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.weights = np.random.randn(layers, num_qubits)
        self.qnode = self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> float:
            # Feature map (Ry rotations)
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)
            # Entanglement
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Variational layers
            for l in range(self.layers):
                for i in range(self.num_qubits):
                    qml.RX(weights[l, i], wires=i)
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))
        return circuit

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return expectation values for a batch of inputs."""
        return np.array([self.qnode(x, self.weights) for x in X])

    def train(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 200) -> None:
        """Simple gradient‑descent training loop."""
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for _ in range(epochs):
            loss = np.mean((self.predict(X) - y) ** 2)
            grads = opt.compute_gradient(
                lambda w: np.mean((self.predict(X, w) - y) ** 2), self.weights
            )
            self.weights = opt.apply_gradients(self.weights, grads)
