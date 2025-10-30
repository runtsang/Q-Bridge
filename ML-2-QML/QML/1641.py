"""Hybrid variational quanvolution layer using PennyLane."""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class AdvancedConv:
    """
    Variational quanvolution layer.
    Supports trainable parameters, adaptive thresholding, and batched execution.
    """

    def __init__(self, kernel_size: int = 2, shots: int = 100,
                 threshold: float = 0.5, adaptive: bool = True) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.adaptive = adaptive
        self.device = qml.device("default.qubit",
                                 wires=self.n_qubits,
                                 shots=shots)
        # Trainable rotation angles
        self.params = pnp.random.randn(self.n_qubits) * 0.01

    @qml.qnode(device=qml.device("default.qubit",
                                 wires=None, shots=None))
    def _circuit(self, data: np.ndarray, params: np.ndarray) -> list[float]:
        """Variational circuit for a single sample."""
        # Data encoding: rotate each qubit by data*π
        for i in range(self.n_qubits):
            qml.RY(data[i] * np.pi, wires=i)
        # Parameterized RX rotations
        for i in range(self.n_qubits):
            qml.RX(params[i], wires=i)
        # Entangling layer
        for i in range(0, self.n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
        # Return expectation values of PauliZ on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of data.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (batch, kernel_size, kernel_size) or (kernel_size, kernel_size).

        Returns
        -------
        np.ndarray
            Array of mean activation values for each sample.
        """
        if data.ndim == 2:
            data = data[None]  # add batch dim
        batch_out = []
        for sample in data:
            flat = sample.reshape(-1)
            expvals = self._circuit(flat, self.params)
            probs = 0.5 * (1 - np.array(expvals))  # convert <Z> to P(|1>)
            if self.adaptive:
                self.threshold = probs.mean()
            batch_out.append(probs.mean())
        return np.array(batch_out)

    def loss(self, data: np.ndarray, targets: np.ndarray) -> float:
        """
        Mean squared error loss between circuit output and targets.

        Parameters
        ----------
        data : np.ndarray
            Input data batch.
        targets : np.ndarray
            Target values.

        Returns
        -------
        float
            Scalar loss value.
        """
        preds = self.run(data)
        return ((preds - targets) ** 2).mean()

    def step(self, data: np.ndarray, targets: np.ndarray, lr: float = 0.01) -> None:
        """
        One gradient descent step using the parameter‑shift rule.

        Parameters
        ----------
        data : np.ndarray
            Input data batch.
        targets : np.ndarray
            Target values.
        lr : float
            Learning rate.
        """
        grad_fn = qml.grad(self.loss)
        grads = grad_fn(data, targets)
        self.params -= lr * grads

def Conv() -> AdvancedConv:
    """
    Factory function to preserve API compatibility with the original Conv().
    Returns an instance of AdvancedConv with default parameters.
    """
    return AdvancedConv()

__all__ = ["AdvancedConv", "Conv"]
