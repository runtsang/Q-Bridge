"""Quantum fully‑connected layer using Pennylane.

The original implementation used a Qiskit circuit with a single
parameterised rotation.  This version expands to a variational
ansatz that can be trained with gradient‑based optimisers, supports
multiple qubits, and returns the expectation value of the Pauli‑Z
operator on the first qubit.  The public ``run`` method mimics the
seed: it accepts an iterable of parameters and outputs a NumPy array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import numpy as np


@dataclass
class FullyConnectedLayer:
    """Parameterized quantum circuit acting as a fully‑connected layer."""
    n_qubits: int = 1
    dev: qml.Device = qml.device("default.qubit", wires=1)
    n_layers: int = 2
    lr: float = 0.01
    epochs: int = 100

    def __post_init__(self) -> None:
        self.params = np.random.uniform(0, 2 * np.pi, size=(self.n_layers, self.n_qubits))
        self._build_qnode()

    def _build_qnode(self) -> None:
        @qml.qnode(self.dev, interface="autograd")
        def circuit(theta: np.ndarray) -> float:
            for layer in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.RY(theta[layer, q], wires=q)
                if layer < self.n_layers - 1:
                    for q in range(self.n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
            return qml.expval(qml.PauliZ(0))

        self.qnode = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the circuit for a single set of parameters."""
        theta = np.array(thetas).reshape(self.n_layers, self.n_qubits)
        expectation = self.qnode(theta)
        return np.array([expectation])

    def train(
        self,
        thetas: Sequence[float],
        targets: Sequence[float],
        lr: float | None = None,
        epochs: int | None = None,
    ) -> Sequence[float]:
        """Train the variational circuit using gradient descent."""
        if lr is not None:
            self.lr = lr
        if epochs is not None:
            self.epochs = epochs

        opt = qml.GradientDescentOptimizer(stepsize=self.lr)

        for _ in range(self.epochs):
            def cost(theta):
                exp = self.qnode(theta)
                return (exp - targets[0]) ** 2  # simple MSE for single target

            self.params = opt.step(cost, self.params)

        return self.params.reshape(-1).tolist()


__all__ = ["FullyConnectedLayer"]
