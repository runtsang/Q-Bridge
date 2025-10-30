import pennylane as qml
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class QCNNModel(BaseEstimator, ClassifierMixin):
    """
    Hybrid quantum‑classical QCNN implemented with Pennylane.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features (default 8).
    layers : int
        Number of convolution‑pooling layers (default 3).
    learning_rate : float
        Learning rate for gradient descent (default 0.01).
    max_iter : int
        Number of optimisation iterations (default 200).
    seed : int
        Random seed for weight initialisation (default 42).
    """

    def __init__(self,
                 num_qubits: int = 8,
                 layers: int = 3,
                 learning_rate: float = 0.01,
                 max_iter: int = 200,
                 seed: int = 42):
        self.num_qubits = num_qubits
        self.layers = layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.seed = seed
        self._dev = qml.device("default.qubit", wires=self.num_qubits)
        self._weights = None

    def _prepare_qnode(self):
        """Build the Pennylane qnode used for prediction and training."""

        @qml.qnode(self._dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> float:
            # Feature embedding
            qml.templates.AngleEmbedding(inputs, wires=range(self.num_qubits), rotation="Z")

            w_idx = 0
            for _ in range(self.layers):
                # Convolution block: RZ‑RY‑CNOT on adjacent qubits
                for i in range(0, self.num_qubits, 2):
                    qml.RZ(weights[w_idx], wires=i)
                    qml.RY(weights[w_idx + 1], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
                    w_idx += 2

                # Pooling: reset every second qubit to |0⟩
                for i in range(1, self.num_qubits, 2):
                    qml.Reset(wires=i)

            # Final readout on qubit 0
            return qml.expval(qml.PauliZ(0))

        return circuit

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QCNNModel":
        """Train the QCNN using gradient descent on binary cross‑entropy."""
        rng = np.random.default_rng(self.seed)
        num_params = self.layers * self.num_qubits  # 2 params per pair → num_qubits per layer
        self._weights = rng.normal(scale=0.1, size=(num_params,))

        circuit = self._prepare_qnode()

        def loss(weights):
            preds = np.array([circuit(x, weights) for x in X])
            # Convert raw circuit output to probability via sigmoid
            probs = 1 / (1 + np.exp(-preds))
            return -np.mean(y * np.log(probs + 1e-9) + (1 - y) * np.log(1 - probs + 1e-9))

        opt = qml.GradientDescentOptimizer(self.learning_rate)
        params = self._weights
        for _ in range(self.max_iter):
            params, _ = opt.step_and_cost(loss, params)

        self._weights = params
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates for each sample."""
        circuit = self._prepare_qnode()
        probs_raw = np.array([circuit(x, self._weights) for x in X])
        probs = 1 / (1 + np.exp(-probs_raw))
        return np.vstack([1 - probs, probs]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

__all__ = ["QCNNModel"]
