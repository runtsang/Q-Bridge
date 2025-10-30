"""
Quantum variational regression model using Pennylane.
It implements a simple angle‑encoding circuit followed by trainable rotation layers.
"""
import numpy as np
import pennylane as qml

class EstimatorQNN:
    """
    Variational quantum neural network with a modular depth and a
    straightforward training loop using the default autograd interface.
    """
    def __init__(self, n_qubits: int = 2, depth: int = 2) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.dev = qml.Device("default.qubit", wires=n_qubits)
        # initialise weights randomly (depth × qubits × 3 rotation params)
        self.weights = np.random.randn(depth, n_qubits, 3)
        self.qnode = self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> float:
            # data encoding with Ry rotations
            for i, val in enumerate(inputs):
                qml.RY(val, wires=i)

            # variational layers
            for layer in weights:
                for i in range(self.n_qubits):
                    qml.Rot(layer[i, 0], layer[i, 1], layer[i, 2], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            return qml.expval(qml.PauliZ(0))

        return circuit

    def loss(self, batch_inputs: np.ndarray, batch_targets: np.ndarray) -> float:
        preds = [self.qnode(x, self.weights) for x in batch_inputs]
        return np.mean((np.array(preds) - batch_targets) ** 2)

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 200, lr: float = 0.01) -> None:
        """
        Gradient‑descent training of the variational parameters.
        """
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for epoch in range(epochs):
            loss_val = self.loss(X, y)
            grads = qml.grad(self.loss)(self.weights, X, y)
            self.weights = opt.step(grads, self.weights)
            if epoch % 20 == 0:
                print(f"Epoch {epoch} loss {loss_val:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.qnode(x, self.weights) for x in X])

def EstimatorQNN() -> EstimatorQNN:
    """Return an untrained quantum estimator instance."""
    return EstimatorQNN()

__all__ = ["EstimatorQNN"]
