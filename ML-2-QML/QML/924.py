import pennylane as qml
import numpy as np
import torch
from typing import Iterable, Tuple, List

class QuantumClassifierModel:
    """
    Variational quantum classifier that mirrors the classical helper.
    Uses PennyLane's default.qubit simulator and autograd for training.
    Supports data‑encoding, entanglement and a simple measurement scheme.
    """

    def __init__(self,
                 num_qubits: int,
                 depth: int = 2,
                 device: str = "default.qubit",
                 lr: float = 0.01,
                 optimizer: str = "Adam"):
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = qml.device(device, wires=num_qubits)
        self.lr = lr
        self.optimizer_name = optimizer

        # Parameter vector for the variational circuit
        self.weights = np.random.randn(depth, num_qubits)
        self.params = self._flatten_weights(self.weights)

        # Optimizer
        if optimizer == "Adam":
            self.optimizer = qml.AdamOptimizer(stepsize=self.lr)
        else:
            self.optimizer = qml.GradientDescentOptimizer(stepsize=self.lr)

    @staticmethod
    def _flatten_weights(weights: np.ndarray) -> np.ndarray:
        return weights.reshape(-1)

    @staticmethod
    def _unflatten_weights(flat: np.ndarray, depth: int, num_qubits: int) -> np.ndarray:
        return flat.reshape((depth, num_qubits))

    @staticmethod
    def build_classifier_circuit(num_qubits: int,
                                 depth: int) -> Tuple[qml.QNode, Iterable, Iterable, List[qml.operation.Operator]]:
        """
        Build a PennyLane QNode that mirrors the quantum helper.
        Returns the QNode, encoding vector, weight vector, and observable list.
        """
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(x: torch.Tensor, weights: torch.Tensor):
            # Data encoding
            for i in range(num_qubits):
                qml.RX(x[i], wires=i)

            # Variational layers
            idx = 0
            for _ in range(depth):
                for i in range(num_qubits):
                    qml.RY(weights[idx], wires=i)
                    idx += 1
                # Entanglement
                for i in range(num_qubits - 1):
                    qml.CZ(wires=[i, i + 1])

            # Measurement
            return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(num_qubits)])

        encoding = torch.arange(num_qubits, dtype=torch.float32)
        weights = torch.randn(depth * num_qubits, dtype=torch.float32)
        observables = [qml.PauliZ(i) for i in range(num_qubits)]
        return circuit, encoding, weights, observables

    def _variational_circuit(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Execute the variational circuit and return expectation values.
        """
        # Data encoding
        for i in range(self.num_qubits):
            qml.RX(x[i], wires=i)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for i in range(self.num_qubits):
                qml.RY(weights[idx], wires=i)
                idx += 1
            for i in range(self.num_qubits - 1):
                qml.CZ(wires=[i, i + 1])

        return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)])

    def fit(self,
            X: torch.Tensor,
            y: torch.Tensor,
            epochs: int = 200,
            batch_size: int = 32,
            verbose: bool = False) -> None:
        """
        Train the variational circuit using stochastic gradient descent.
        """
        X = X.to(torch.float32)
        y = y.to(torch.long)

        @qml.qnode(self.device, interface="torch")
        def circuit(x, weights):
            return self._variational_circuit(x, weights)

        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            perm = torch.randperm(X.size(0))
            epoch_loss = 0.0

            for i in range(0, X.size(0), batch_size):
                idx = perm[i:i + batch_size]
                xb = X[idx]
                yb = y[idx]

                self.params = self.optimizer.step(lambda w: loss_fn(
                    torch.log_softmax(circuit(xb, w), dim=1), yb), self.params)

                logits = torch.log_softmax(circuit(xb, self.params), dim=1)
                batch_loss = loss_fn(logits, yb).item()
                epoch_loss += batch_loss * xb.size(0)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss / X.size(0):.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return class predictions using the trained variational circuit.
        """
        @qml.qnode(self.device, interface="torch")
        def circuit(x, weights):
            return self._variational_circuit(x, weights)

        with torch.no_grad():
            logits = torch.log_softmax(circuit(X, self.params), dim=1)
            return torch.argmax(logits, dim=1)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.predict(X)

__all__ = ["QuantumClassifierModel"]
