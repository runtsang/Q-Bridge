"""Variational quantum circuit that emulates a fully‑connected layer."""

import numpy as np
import pennylane as qml
from typing import Iterable, List, Tuple

class FullyConnectedLayer:
    """Variational quantum circuit that emulates a fully‑connected layer.

    Parameters are passed as a flat list.  The circuit consists of
    ``n_layers`` blocks of single‑qubit rotations followed by a
    linear‑entangling layer.  The expectation value of the Pauli‑Z
    operator on the first qubit is returned, mirroring the scalar
    output of the classical counterpart.
    """
    def __init__(
        self,
        n_qubits: int = 1,
        n_layers: int = 2,
        dev: str = "default.qubit"
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(dev, wires=n_qubits)
        self.n_params = n_qubits * n_layers * 3  # Ry,Rz,Rx per layer

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            # params shape: (n_layers, n_qubits, 3)
            for layer in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.RY(params[layer, q, 0], wires=q)
                    qml.RZ(params[layer, q, 1], wires=q)
                    qml.RX(params[layer, q, 2], wires=q)
                # linear entanglement
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        # initialise parameters randomly
        self.params = np.random.randn(self.n_layers, self.n_qubits, 3)

    def run(self, theta: Iterable[float]) -> np.ndarray:
        """Evaluate the circuit for a flattened parameter list."""
        params = np.array(theta).reshape(self.n_layers, self.n_qubits, 3)
        expval = self.circuit(params)
        return np.array([expval])

    def set_parameters(self, theta: Iterable[float]) -> None:
        self.params = np.array(theta).reshape(self.n_layers, self.n_qubits, 3)

    def train_on_synthetic(
        self,
        epochs: int = 30,
        lr: float = 0.01,
        seed: int = 42
    ) -> Tuple[np.ndarray, List[float]]:
        """Simple training loop on synthetic data using the parameter‑shift rule.

        Returns the final flattened parameters and a list of training losses.
        """
        np.random.seed(seed)
        # Synthetic regression: y = sin(x)
        xs = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
        ys = np.sin(xs)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in zip(xs, ys):
                # For each sample, evaluate circuit with current params
                expval = self.run(self.params.flatten())[0]
                loss = (expval - y) ** 2
                epoch_loss += loss

                # Gradient via parameter‑shift rule
                grads = qml.gradients.param_shift(self.circuit)(self.params)
                self.params -= lr * grads
            losses.append(epoch_loss / len(xs))
        return self.params.flatten(), losses

def FCL() -> FullyConnectedLayer:
    """Convenience factory matching the original API."""
    return FullyConnectedLayer()

__all__ = ["FullyConnectedLayer", "FCL"]
