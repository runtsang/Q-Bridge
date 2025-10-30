import pennylane as qml
import numpy as np

class EstimatorQNNExtended:
    """
    Variational quantum regression model.

    Two‑qubit ansatz with parameterised rotations and a CNOT entanglement chain.
    Input is encoded via Ry rotations on each qubit and the output is the
    expectation value of Pauli‑Y on qubit 0.
    """
    def __init__(self,
                 n_qubits: int = 2,
                 n_layers: int = 2,
                 dev: qml.Device | None = None) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)

        # Trainable weights: shape (n_layers, n_qubits, 3) for Rot(α,β,γ)
        self.weights = np.random.randn(n_layers, n_qubits, 3)

        # Create a QNode that accepts input vector and flattened weights
        def _quantum_forward(x: np.ndarray, flat_w: np.ndarray):
            # Reshape weights back to (n_layers, n_qubits, 3)
            w = flat_w.reshape(self.n_layers, self.n_qubits, 3)
            # Input encoding
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            # Variational layers
            for layer in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.Rot(*w[layer, q], wires=q)
                # Entangling CNOT chain
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return qml.expval(qml.PauliY(0))
        self.qnode = qml.QNode(_quantum_forward, self.dev, interface="autograd")

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the current model on a single input vector."""
        return self.qnode(x, self.weights.reshape(-1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vectorised prediction over a batch of inputs (shape: [batch, n_qubits])."""
        return np.array([self(x) for x in X])

__all__ = ["EstimatorQNNExtended"]
