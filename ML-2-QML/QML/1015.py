import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer

class EstimatorQNN:
    """
    Variational quantum circuit for regression.
    Implements data‑encoding via RY gates, a stack of parameterized rotation layers,
    simple CNOT entanglement, and estimates the expectation of a Pauli‑Y observable.
    The interface mirrors the classical EstimatorQNN (fit, predict, score).
    """
    def __init__(self,
                 n_qubits: int = 1,
                 n_layers: int = 3,
                 dev: qml.Device | None = None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = dev if dev is not None else qml.device("default.qubit", wires=n_qubits)
        self.params = pnp.random.randn(n_layers, n_qubits, 3)
        self.obs = qml.PauliY(0)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, params):
            # data encoding
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            # variational layers
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(*params[layer, i], wires=i)
                # simple entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(self.obs)

        self.circuit = circuit

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for x in X:
            preds.append(self.circuit(x, self.params))
        return np.array(preds)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 200,
            lr: float = 0.01,
            verbose: bool = False) -> None:
        opt = AdamOptimizer(lr)
        for epoch in range(1, epochs + 1):
            def loss_fn(params):
                preds = self.predict(X)
                return np.mean((preds - y) ** 2)

            self.params, _ = opt.step_and_cost(loss_fn, self.params)
            if verbose and epoch % 20 == 0:
                loss = loss_fn(self.params)
                print(f"Epoch {epoch:03d} – loss: {loss:.6f}")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))

__all__ = ["EstimatorQNN"]
