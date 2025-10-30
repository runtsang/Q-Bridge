"""Quantum classifier implementation using Pennylane."""
import pennylane as qml
import numpy as np
from typing import Tuple, List

class QuantumClassifier:
    """
    A variational quantum circuit that outputs a binary decision via expectation values.
    The circuit consists of an encoding layer, a configurable number of entangling layers,
    and a measurement of Z on each qubit. The outputs can be used as logits for a binary
    classification task.
    """
    def __init__(self, num_qubits: int, depth: int = 1, device: str = "default.qubit"):
        self.num_qubits = num_qubits
        self.depth = depth
        self.dev = qml.device(device, wires=num_qubits)
        self._build_circuit()

    def _build_circuit(self) -> None:
        @qml.qnode(self.dev)
        def circuit(inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
            # Feature encoding
            for i in range(self.num_qubits):
                qml.RX(inputs[i], wires=i)

            # Variational layers
            idx = 0
            for _ in range(self.depth):
                for i in range(self.num_qubits):
                    qml.RY(params[idx], wires=i)
                    idx += 1
                for i in range(self.num_qubits - 1):
                    qml.CZ(wires=[i, i + 1])

            # Observables: Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.qnode = circuit

    def forward(self, inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Compute expectation values for a single data point.
        """
        return self.qnode(inputs, params)

    def get_initial_params(self) -> np.ndarray:
        """
        Randomly initialise variational parameters.
        """
        rng = np.random.default_rng()
        return rng.normal(size=self.num_qubits * self.depth)

    def loss(self, inputs: np.ndarray, params: np.ndarray, target: int) -> float:
        """
        Binary cross‑entropy loss between logits derived from expectation values and the target label.
        """
        preds = self.forward(inputs, params)
        logits = np.array(preds)
        # Map Z expectation [-1,1] to logits via tanh
        logits = 0.5 * (logits + 1.0)
        # Binary cross‑entropy
        p = logits[0]  # use first qubit as output
        return -(target * np.log(p + 1e-9) + (1 - target) * np.log(1 - p + 1e-9))

    def train(self, data: np.ndarray, labels: np.ndarray, lr: float = 0.01,
              epochs: int = 100, batch_size: int = 32) -> np.ndarray:
        """
        Gradient‑based training of the variational parameters using Pennylane's autograd.
        """
        params = self.get_initial_params()
        opt = qml.GradientDescentOptimizer(lr)

        for _ in range(epochs):
            perm = np.random.permutation(len(data))
            for i in range(0, len(data), batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_data = data[batch_idx]
                batch_labels = labels[batch_idx]

                def loss_fn(p):
                    loss = 0.0
                    for x, y in zip(batch_data, batch_labels):
                        loss += self.loss(x, p, y)
                    return loss / len(batch_data)

                params = opt.step(loss_fn, params)
        return params

__all__ = ["QuantumClassifier"]
