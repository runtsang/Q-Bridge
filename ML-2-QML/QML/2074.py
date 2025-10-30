"""Quantum convolutional filter implemented with PennyLane.

Features:
- Parameterized variational circuit with RX and CZ entanglement
- Data encoded via angle embedding of pixel values
- Expectation value of PauliZ on all qubits averaged
- Supports hybrid training via autograd
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from typing import Tuple


def Conv() -> qml.QNode:
    """Return a callable object that emulates a quantum convolutional filter."""

    class QuanvCircuit:
        def __init__(
            self,
            kernel_size: int = 2,
            dev: qml.Device | None = None,
            threshold: float = 0.5,
            num_layers: int = 2,
        ) -> None:
            self.kernel_size = kernel_size
            self.n_qubits = kernel_size ** 2
            self.threshold = threshold
            self.num_layers = num_layers

            self.dev = dev or qml.device("default.qubit", wires=self.n_qubits)

            # initialise trainable parameters
            self.params = pnp.random.uniform(0, 2 * np.pi, (num_layers, self.n_qubits))

        @qml.qnode
        def circuit(self, data: np.ndarray, params: np.ndarray) -> float:
            """Variational circuit that processes a single kernel."""
            # angle embedding of the classical pixel values
            for i in range(self.n_qubits):
                qml.RY(np.pi * data[i], wires=i)

            # variational layers
            for layer in range(self.num_layers):
                for i in range(self.n_qubits):
                    qml.RX(params[layer, i], wires=i)
                # entanglement
                for i in range(self.n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
                qml.CZ(wires=[self.n_qubits - 1, 0])

            # measurement: average PauliZ expectation over all qubits
            return sum(qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)) / self.n_qubits

        def run(self, data: np.ndarray) -> float:
            """Run the circuit on a 2‑D kernel and return the output."""
            flat = data.reshape(self.n_qubits)
            return self.circuit(flat, self.params)

        def gradient(self, data: np.ndarray) -> np.ndarray:
            """Return the gradient of the output w.r.t. the trainable parameters."""
            flat = data.reshape(self.n_qubits)
            return qml.grad(self.circuit)(flat, self.params)

    return QuanvCircuit()
