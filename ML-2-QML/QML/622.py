"""Quantum convolutional filter using Pennylane and a Qiskit Aer backend.

The circuit applies an RX rotation to each qubit, followed by a random
entangling layer (CRX gates).  Parameters are bound to the input image
by setting the rotation angle to π when the pixel value exceeds a
threshold.  The output is the mean probability of measuring |1> across
all qubits.  The implementation supports gradient computation via
Pennylane's parameter‑shift rule, enabling end‑to‑end training on a
quantum simulator or real device.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from qiskit import Aer


def Conv() -> "QuantumConvFilter":
    """Return a callable object that emulates the quantum filter."""
    return QuantumConvFilter()


class QuantumConvFilter:
    """
    Variational quantum convolution filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Number of qubits (kernel_size^2).
    threshold : float, default 0.5
        Pixel threshold for setting the RX rotation angle.
    shots : int, default 1024
        Number of shots for expectation estimation.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.5, shots: int = 1024) -> None:
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots

        # Create a Qiskit Aer device that Pennylane can use
        self.dev = qml.device("qiskit.aer", wires=self.n_qubits, shots=self.shots)

        # Define the variational circuit as a QNode
        @qml.qnode(self.dev, interface="torch", diff_method="parameter_shift")
        def circuit(params: np.ndarray, data: np.ndarray) -> np.ndarray:
            # Apply data‑dependent RX rotations
            for i in range(self.n_qubits):
                angle = np.pi if data[i] > self.threshold else 0.0
                qml.RX(angle, wires=i)

            # Random entangling layer (CRX with random angles)
            for i in range(self.n_qubits - 1):
                qml.CRX(np.pi / 4, wires=[i, i + 1])

            # Parameterized rotation layer
            for i in range(self.n_qubits):
                qml.RX(params[i], wires=i)

            # Measurement: expectation of PauliZ on all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

        # Initialize trainable parameters
        self.params = np.random.uniform(0, 2 * np.pi, size=self.n_qubits)

    def run(self, data: np.ndarray) -> float:
        """
        Evaluate the quantum filter on a single 2‑D image patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean probability of measuring |1> across all qubits.
        """
        flat = data.reshape(-1)
        expvals = self.circuit(self.params, flat)
        # Convert PauliZ expectation to probability of |1>
        probs = 0.5 * (1 - np.array(expvals))
        return probs.mean()

    def train_step(self, data: np.ndarray, target: float, lr: float = 0.01) -> float:
        """
        Perform a single gradient‑descent step on the parameters.

        Parameters
        ----------
        data : np.ndarray
            Input patch.
        target : float
            Desired output value.
        lr : float
            Learning rate.

        Returns
        -------
        float
            Loss value after the update.
        """
        # Compute loss (MSE)
        pred = self.run(data)
        loss = (pred - target) ** 2

        # Compute gradients via parameter‑shift
        grads = np.gradient(loss, self.params)

        # Update parameters
        self.params -= lr * grads
        return loss

    def parameters(self) -> np.ndarray:
        """Return the current trainable parameters."""
        return self.params

    def set_parameters(self, params: np.ndarray) -> None:
        """Set the trainable parameters."""
        self.params = np.array(params, copy=True)
