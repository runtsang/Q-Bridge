import pennylane as qml
import pennylane.numpy as np
import numpy as onp
from typing import Iterable, Tuple

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[object, Iterable, Iterable, list]:
    """
    Create a data‑re‑uploading ansatz that mirrors the classical pipeline but
    uses a variational circuit.  It returns the QNode, encoding parameters,
    trainable weights and the list of Pauli‑Z observables.
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(x_vals, weight_vals):
        # Initial data encoding – one RX per qubit
        for i in range(num_qubits):
            qml.RX(x_vals[i], wires=i)

        # Variational layers with re‑upload
        idx = 0
        for _ in range(depth):
            for i in range(num_qubits):
                qml.RY(weight_vals[idx], wires=i)
                idx += 1
            # Entangling layer
            for i in range(num_qubits - 1):
                qml.CZ(wires=[i, i + 1])

        # Measure Z on the first qubit as a binary readout
        return qml.expval(qml.PauliZ(0))

    encoding = list(range(num_qubits))
    weight_sizes = list(range(num_qubits * depth))
    observables = [qml.PauliZ(0)]

    return circuit, encoding, weight_sizes, observables

class HybridQuantumClassifier:
    """
    Quantum‑enhanced classifier that implements a full training loop with
    PennyLane's autograd backend.  It exposes the same metadata interface
    (encoding, weight sizes, observables) as the classical counterpart.
    """
    def __init__(self,
                 num_qubits: int,
                 depth: int = 3,
                 lr: float = 0.01,
                 epochs: int = 200,
                 batch_size: int = 32,
                 device: str = "default.qubit"):
        self.circuit, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_qubits, depth)
        self.optimizer = qml.GradientDescentOptimizer(stepsize=lr)
        self.epochs = epochs
        self.batch_size = batch_size

    def _loss(self, x_batch, y_batch):
        preds = np.array([self.circuit(x, self.weights) for x in x_batch])
        logits = 0.5 * (preds + 1.0)  # map from [-1,1] to [0,1]
        eps = 1e-7
        return np.mean(-y_batch * np.log(logits + eps) - (1 - y_batch) * np.log(1 - logits + eps))

    def fit(self, X: onp.ndarray, y: onp.ndarray) -> None:
        """
        Train the circuit using stochastic gradient descent.
        """
        self.weights = np.random.randn(len(self.weight_sizes)) * 0.1

        num_samples = X.shape[0]
        for epoch in range(self.epochs):
            indices = np.random.permutation(num_samples)
            epoch_loss = 0.0
            for start in range(0, num_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                x_batch = X[batch_idx]
                y_batch = y[batch_idx]
                self.weights, batch_loss = self.optimizer.step_and_cost(
                    self.weights, lambda w: self._loss(x_batch, y_batch)
                )
                epoch_loss += batch_loss
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}  loss={epoch_loss / (num_samples / self.batch_size):.4f}")

    def predict(self, X: onp.ndarray) -> onp.ndarray:
        preds = np.array([self.circuit(x, self.weights) for x in X])
        probs = 0.5 * (preds + 1.0)
        return (probs > 0.5).astype(int)

    def evaluate(self, X: onp.ndarray, y: onp.ndarray) -> float:
        preds = self.predict(X)
        return (preds == y).mean()

    def get_weight_sizes(self) -> Iterable[int]:
        return self.weight_sizes

    def get_encoding(self) -> Iterable[int]:
        return self.encoding

    def get_observables(self) -> list:
        return self.observables
