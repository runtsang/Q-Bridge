import pennylane as qml
import pennylane.numpy as np
from pennylane import numpy as pnp

dev = qml.device("default.qubit", wires=3)

def param_embedding(x, weights):
    """Embed classical data into rotation angles."""
    qml.AngleEmbedding(x, wires=[0, 1])
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)

def variational_circuit(x, weights):
    """Two‑qubit entangling variational circuit."""
    param_embedding(x, weights[:2])
    qml.CNOT(wires=[0, 1])
    qml.RZ(weights[2], wires=0)
    qml.RZ(weights[3], wires=1)
    qml.CNOT(wires=[1, 2])
    qml.RZ(weights[4], wires=2)
    return qml.expval(qml.PauliZ(0))

class EstimatorQNN:
    """Hybrid quantum neural network using Pennylane."""
    def __init__(self, n_weights: int = 5):
        self.n_weights = n_weights
        self.weights = np.random.uniform(-np.pi, np.pi, size=n_weights)

    def predict(self, x: np.ndarray) -> float:
        return qml.execute([variational_circuit],
                           [dev],
                           [self.weights],
                           [x])[0]

    def batch_predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict(x) for x in X])

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.batch_predict(X)
        return np.mean((preds - y) ** 2)

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              lr: float = 0.01,
              epochs: int = 200):
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for _ in range(epochs):
            self.weights = opt.step(lambda w: self.loss(X, y), self.weights)
        return self.weights

def EstimatorQNN() -> EstimatorQNN:
    """Return an instance of the variational QNN."""
    return EstimatorQNN()

__all__ = ["EstimatorQNN"]
