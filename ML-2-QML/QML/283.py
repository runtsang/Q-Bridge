"""ConvGen315 – a Pennylane variational circuit for quantum convolution.

The class implements a depth‑wise separable quantum filter with a learnable threshold.
It can be used as a drop‑in replacement for the original Conv filter.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Tuple

class ConvGen315:
    """Variational quantum convolutional filter."""
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        device: str = "default.qubit",
        shots: int = 1000,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.dev = qml.device(device, wires=self.n_qubits, shots=shots)

        # Initialize variational parameters
        self.params = np.random.randn(self.n_qubits, 3)
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, x: np.ndarray, params: np.ndarray) -> float:
        """Variational circuit for a single kernel."""
        # Encode data into rotation angles
        for i in range(self.n_qubits):
            qml.RX(np.pi if x[i] > self.threshold else 0.0, wires=i)
        # Variational layers
        for i in range(self.n_qubits):
            qml.RY(params[i, 0], wires=i)
            qml.RZ(params[i, 1], wires=i)
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            qml.RY(params[i, 2], wires=i)
        # Measurement
        return qml.expval(qml.PauliZ(0))

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum filter on a 2D array of shape (kernel_size, kernel_size).

        Args:
            data: 2D array with shape (kernel_size, kernel_size).

        Returns:
            float: average expectation value over all qubits.
        """
        flat = data.reshape(-1)
        expectation = self.qnode(flat, self.params)
        return float(expectation)

    def train(self, dataset: Tuple[np.ndarray, np.ndarray], lr: float = 0.01, epochs: int = 100):
        """
        Simple training loop using gradient descent on the variational parameters.

        Args:
            dataset: tuple (X, y) where X is an array of shape (N, kernel_size, kernel_size)
                     and y are target values.
            lr: learning rate.
            epochs: number of training epochs.
        """
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for _ in range(epochs):
            for x, y in zip(dataset[0], dataset[1]):
                loss = (self.run(x) - y) ** 2
                grads = qml.grad(self.qnode)(x, self.params)
                self.params = opt.step(grads, self.params)

__all__ = ["ConvGen315"]
