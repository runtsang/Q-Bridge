"""Quantum implementation of the Conv filter using Pennylane.

The circuit is a parameter‑efficient ansatz that operates on a
square block of size ``kernel_size``.  The input data is encoded
by applying an X rotation whose angle depends on the pixel value
relative to a threshold.  The circuit is executed on the
``default.qubit`` simulator and returns the mean expectation
value of the Z observable across all qubits.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from typing import Optional


class Conv:
    """
    Args:
        kernel_size (int): Size of the convolution kernel.
        threshold (float): Pixel value threshold for encoding.
        shots (int): Number of shots for the simulator.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        shots: int = 100,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.n_qubits = kernel_size ** 2

        # Define a device
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)

        # Build a parameter‑efficient ansatz
        self._build_ansatz()

    def _build_ansatz(self) -> None:
        """Construct a simple variational ansatz with one layer of RY gates
        and a single entangling CNOT chain."""
        @qml.qnode(self.dev)
        def circuit(params, data):
            # Data encoding: rotate each qubit by an angle that depends on
            # the pixel value relative to the threshold.
            for i in range(self.n_qubits):
                theta = np.pi if data[i] > self.threshold else 0.0
                qml.RX(theta, wires=i)

            # Variational parameters
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)

            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum circuit on a single 2D patch.

        Args:
            data: 2D array with shape (kernel_size, kernel_size).

        Returns:
            float: Mean expectation value of Z across all qubits,
            rounded to 4 decimal places.
        """
        flat = data.flatten()
        # Initialise variational parameters to zeros for simplicity
        params = np.zeros(self.n_qubits)
        expvals = self.circuit(params, flat)
        mean_expval = np.mean(expvals)
        return float(np.round(mean_expval, 4))

    def __repr__(self) -> str:
        return (
            f"Conv(kernel_size={self.kernel_size}, threshold={self.threshold}, "
            f"shots={self.shots})"
        )


__all__ = ["Conv"]
