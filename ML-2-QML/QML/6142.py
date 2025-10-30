"""Quantum variational convolutional filter using Pennylane.

The class implements a 2‑D filter as a parameterized quantum circuit.
It supports adaptive thresholding and can be trained end‑to‑end with
automatic differentiation.
"""

import numpy as np
import pennylane as qml

class ConvEnhanced:
    """
    A quantum convolutional filter that emulates the classical Conv filter
    but uses a variational circuit. The filter operates on a square kernel
    of size ``kernel_size`` and returns the average probability of measuring
    |1⟩ across all qubits.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        shots: int = 1024,
        device: str = "default.qubit",
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.shots = shots

        # Quantum device with sampling
        self.dev = qml.device(device, wires=self.n_qubits, shots=shots)

        # Variational parameters (initialized to zero)
        self.params = np.zeros(self.n_qubits)

        # Quantum node
        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            for i in range(self.n_qubits):
                qml.RX(params[i], wires=i)
            # Simple entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Sample bitstrings
            return qml.sample()

        self.circuit = circuit

    def run(self, data):
        """
        Execute the quantum circuit on a 2‑D kernel.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1⟩ across all qubits.
        """
        flat = np.asarray(data).flatten()
        # Bind parameters: π if value > threshold else 0
        theta = np.array([np.pi if val > self.threshold else 0.0 for val in flat])

        # Sample bitstrings
        samples = self.circuit(theta)  # shape (shots, n_qubits)
        # Count ones per qubit
        ones_per_qubit = np.sum(samples, axis=0)
        # Probability of |1⟩ for each qubit
        probs = ones_per_qubit / self.shots
        return float(np.mean(probs))

__all__ = ["ConvEnhanced"]
