import pennylane as qml
import numpy as np
from pennylane.optimize import AdamOptimizer

class QCNNHybrid:
    """
    Quantum circuit implementing a QCNN‑style ansatz with convolution and pooling layers.
    The circuit is parameter‑shift differentiable and can be trained with a simple
    Adam optimizer. The final observable is the expectation value of PauliZ on the
    last qubit, which serves as a scalar output for binary classification.
    """
    def __init__(self, num_qubits: int = 8, feature_dim: int = 8,
                 device: str = "default.qubit"):
        self.num_qubits = num_qubits
        self.feature_dim = feature_dim
        self.dev = qml.device(device, wires=num_qubits)
        # Feature‑map parameters (here we use a simple RZ rotation per input feature)
        self.feature_params = np.zeros(feature_dim)
        # Ansatz parameters: 3 parameters per two‑qubit block
        # Number of two‑qubit blocks = num_qubits // 2
        self.ansatz_params = np.zeros((num_qubits // 2, 3))
        # Build the QNode
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, x, feature_params, ansatz_params):
        # Feature map: encode classical data into rotations
        for i, wire in enumerate(range(self.num_qubits)):
            qml.RZ(x[i], wires=wire)
        # Convolution + pooling ansatz
        # Two‑qubit blocks
        for block_idx, (w1, w2, w3) in enumerate(ansatz_params):
            q1, q2 = 2*block_idx, 2*block_idx+1
            # Convolution pattern inspired by the seed
            qml.CZ(q1, q2)
            qml.RZ(w1, wires=q1)
            qml.RY(w2, wires=q2)
            qml.CZ(q1, q2)
            qml.RZ(w3, wires=q2)
        # Pooling: measure Z on the last qubit
        return qml.expval(qml.PauliZ(self.num_qubits - 1))

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the circuit on a single data point."""
        return self.qnode(x, self.feature_params, self.ansatz_params)

    def fit(self, data: np.ndarray, targets: np.ndarray,
            epochs: int = 50, lr: float = 0.01, verbose: bool = False) -> dict:
        """
        Train the ansatz parameters using the Adam optimizer and the parameter‑shift
        rule. Returns a history of the mean‑squared‑error loss.
        """
        opt = AdamOptimizer(stepsize=lr)
        history = {"mse_loss": []}
        for epoch in range(epochs):
            # Compute loss and gradient over the whole batch
            def loss_fn(params):
                preds = np.array([self.qnode(x, self.feature_params, params) for x in data])
                return np.mean((preds - targets)**2)
            grads = qml.grad(loss_fn)(self.ansatz_params)
            self.ansatz_params = opt.step(grads, self.ansatz_params)
            history["mse_loss"].append(loss_fn(self.ansatz_params))
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - MSE: {history['mse_loss'][-1]:.4f}")
        return history

__all__ = ["QCNNHybrid"]
