"""QuantumClassifierModel: Variational circuit with parameter shift gradient.

Features:
* Angle encoding of classical data
* Flexible ansatz depth and entanglement patterns
* Support for Pennylane backends
* Gradient via parameter shift (exact for Pauli observables)
* Metadata: encoding parameters, weight parameters, observables
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Callable

import pennylane as qml
import numpy as np

class QuantumClassifierModel:
    """Variational quantum classifier with metadata mirroring the classical side."""
    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        entanglement: str = "circular",
        backend: str = "default.qubit",
        device: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        num_qubits: int
            Number of qubits / features.
        depth: int
            Number of ansatz layers.
        entanglement: str
            Entanglement pattern for each layer: "circular", "full", or "none".
        backend: str
            Pennylane device name (e.g., 'default.qubit', 'aer').
        device: str | None
            Explicit device name; if None, uses ``backend``.
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.dev = qml.device(device or backend, wires=num_qubits)

        # Parameter vectors
        self.encoding_params = np.arange(num_qubits, dtype=float)
        self.weight_params = np.arange(num_qubits * depth, dtype=float)

        self.observables = [qml.PauliZ(i) for i in range(num_qubits)]

    def circuit(self, x: np.ndarray, params: np.ndarray) -> None:
        """Variational circuit with angle encoding."""
        # Encode data
        for i, val in enumerate(x):
            qml.RX(val, wires=i)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for i in range(self.num_qubits):
                qml.RY(params[idx], wires=i)
                idx += 1
            # Entangling gates
            if self.entanglement == "circular":
                for i in range(self.num_qubits):
                    qml.CZ(wires=[i, (i + 1) % self.num_qubits])
            elif self.entanglement == "full":
                for i in range(self.num_qubits):
                    for j in range(i + 1, self.num_qubits):
                        qml.CZ(wires=[i, j])
            # else "none": no entanglement

    def expectation(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Return expectation values of the observables for input x."""
        @qml.qnode(self.dev, interface="autograd")
        def qnode(x, params):
            self.circuit(x, params)
            return [qml.expval(obs) for obs in self.observables]

        return qnode(x, params)

    def parameter_shift_gradient(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Compute gradient via parameter‑shift rule."""
        shift = np.pi / 2
        grad = np.zeros_like(params)

        for i in range(len(params)):
            shift_vec = np.zeros_like(params)
            shift_vec[i] = shift

            f_plus = self.expectation(x, params + shift_vec).sum()
            f_minus = self.expectation(x, params - shift_vec).sum()
            grad[i] = (f_plus - f_minus) / 2.0

        return grad

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> None:
        """Stochastic gradient descent on the expectation‑based loss."""
        params = np.random.randn(self.num_qubits * self.depth)
        loss_fn = lambda p, x, t: (self.expectation(x, p) @ t)  # simple linear readout

        for epoch in range(1, epochs + 1):
            loss = 0.0
            for x, t in zip(X, y):
                grad = self.parameter_shift_gradient(x, params)
                params -= lr * grad
                loss += loss_fn(params, x, t)
            loss /= len(X)
            if verbose:
                print(f"Epoch {epoch:3d} loss={loss:.4f}")

        self.params = params

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (0 or 1)."""
        preds = []
        for x in X:
            exps = self.expectation(x, self.params)
            preds.append(int(np.argmax(exps)))
        return np.array(preds)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy."""
        preds = self.predict(X)
        return (preds == y).mean()

    def get_metadata(self) -> Tuple[Iterable[int], Iterable[int], List[int]]:
        """Return encoding indices, weight indices, and observables indices."""
        return self.encoding_params, self.weight_params, list(range(self.num_qubits))

__all__ = ["QuantumClassifierModel"]
