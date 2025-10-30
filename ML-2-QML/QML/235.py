"""Variational quantum regressor using Pennylane.

Features
--------
- Multi‑qubit circuit with alternating rotation and entanglement layers.
- Customisable entanglement pattern (full, nearest‑neighbour, or none).
- Built‑in training routine that optimises expectation value of Pauli‑Z
  on the last qubit using a classical optimiser.
"""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
from pennylane import numpy as pnp
from typing import Sequence, Callable

__all__ = ["EstimatorQNN"]


class EstimatorQNN:
    """Quantum neural network for regression.

    Parameters
    ----------
    num_qubits: int
        Number of qubits in the circuit.
    layers: int
        Number of variational layers.
    entanglement: str or Sequence[tuple[int, int]]
        Entanglement pattern.  Supported strings: 'full', 'nearest', 'none'.
        Alternatively provide a custom list of qubit pairs.
    weight_init: str, default 'normal'
        Weight initialization scheme for rotation angles.
    """

    def __init__(
        self,
        num_qubits: int,
        layers: int,
        *,
        entanglement: str | Sequence[tuple[int, int]] = "full",
        weight_init: str = "normal",
    ) -> None:
        self.num_qubits = num_qubits
        self.layers = layers
        self.entanglement = entanglement
        self.weight_init = weight_init

        # build the quantum device
        self.dev = qml.device("default.qubit", wires=self.num_qubits)

        # create the variational circuit
        @qml.qnode(self.dev, interface="autograd")
        def circuit(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # data encoding
            for i in range(self.num_qubits):
                qml.RY(x[i], wires=i)
            # variational layers
            weight_idx = 0
            for _ in range(self.layers):
                for i in range(self.num_qubits):
                    # RX, RZ rotations with trainable parameters
                    qml.RX(weights[weight_idx], wires=i)
                    weight_idx += 1
                    qml.RZ(weights[weight_idx], wires=i)
                    weight_idx += 1
                # entanglement
                if self.entanglement == "full":
                    for i in range(self.num_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[self.num_qubits - 1, 0])
                elif self.entanglement == "nearest":
                    for i in range(self.num_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                elif self.entanglement == "none":
                    pass
                else:
                    # custom list of pairs
                    for (i, j) in self.entanglement:
                        qml.CNOT(wires=[i, j])
            return qml.expval(qml.PauliZ(self.num_qubits - 1))

        self.circuit = circuit

        # initialise trainable weights
        num_weights = 2 * self.num_qubits * self.layers
        if self.weight_init == "normal":
            self.weights = np.random.randn(num_weights)
        else:
            self.weights = np.random.uniform(-np.pi, np.pi, num_weights)

    def predict(self, x: Sequence[float]) -> float:
        """Return the expectation value for a single input vector."""
        x_arr = np.array(x, dtype=np.float64)
        return float(self.circuit(x_arr, self.weights))

    def train(
        self,
        data: Sequence[tuple[Sequence[float], float]],
        lr: float = 0.01,
        epochs: int = 100,
        optimiser: Callable = qml.GradientDescentOptimizer,
    ) -> list[float]:
        """Train the circuit using a simple gradient descent scheme.

        Parameters
        ----------
        data: list of (input, target) tuples
        lr: learning rate for the optimiser
        epochs: number of optimisation steps
        optimiser: constructor for a Pennylane optimiser

        Returns
        -------
        loss_history: list of loss values per epoch
        """
        opt = optimiser(lr=lr)
        loss_history = []

        for _ in range(epochs):
            loss = 0.0
            for x, y in data:
                def cost(weights):
                    pred = self.circuit(np.array(x, dtype=np.float64), weights)
                    return (pred - y) ** 2

                loss += cost(self.weights)
                self.weights = opt.step(cost, self.weights)

            loss_history.append(loss / len(data))
        return loss_history
