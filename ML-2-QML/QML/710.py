import pennylane as qml
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class EstimatorQNN(BaseEstimator, RegressorMixin):
    """
    Quantum neural network regressor based on a parameterised PennyLane ansatz.
    It exposes the scikit‑learn API and can be trained with gradient‑based
    optimisers.  The circuit supports multiple qubits, layers and an
    optional Pauli observable.
    """
    def __init__(self,
                 n_qubits: int = 2,
                 layers: int = 2,
                 use_observable: bool = True):
        self.n_qubits = n_qubits
        self.layers = layers
        self.use_observable = use_observable
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Random initial parameters: (layers, qubits, 3 rotation angles)
        self.params = np.random.randn(layers, n_qubits, 3)
        # Data‑encoding parameters per qubit
        self.input_params = np.random.randn(n_qubits, 3)

        self._qnode = self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Data encoding
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
                qml.RY(self.input_params[i, 1], wires=i)
                qml.RZ(self.input_params[i, 2], wires=i)

            # Parameterised ansatz
            for layer in range(self.layers):
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                # Entanglement
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # Measurement
            if self.use_observable:
                return qml.expval(qml.PauliZ(0))
            else:
                return qml.expval(qml.PauliY(0))
        return circuit

    def _loss(self, X, y, weights):
        preds = np.array([self._qnode(x, weights) for x in X])
        return np.mean((preds - y) ** 2)

    def fit(self,
            X,
            y,
            epochs: int = 200,
            lr: float = 0.01,
            verbose: bool = False):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        weights = self.params

        for epoch in range(epochs):
            grads = np.zeros_like(weights)
            for i in range(len(X)):
                grads += qml.grad(self._qnode)(X[i], weights)
            grads /= len(X)
            weights -= lr * grads
            if verbose and (epoch + 1) % 20 == 0:
                loss = self._loss(X, y, weights)
                print(f"[epoch {epoch+1}] loss={loss:.4f}")

        self.params = weights
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.array([self._qnode(x, self.params) for x in X])

__all__ = ["EstimatorQNN"]
