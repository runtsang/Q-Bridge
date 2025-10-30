"""Quantum‑only variant of ConvEnhanced that uses a Pennylane variational circuit
to extract features from a 2‑D kernel.

The module exposes a single callable class that can be instantiated
with a parameter list.  The class can be used in a classical
pipeline by returning a scalar value that is the expectation value
of a quantum circuit.  The QML implementation is completely
self‑contained and fully compatible with the original seed.
"""

import pennylane as qml
import torch
import numpy as np


class ConvEnhanced:
    """Variational quantum filter for a 2‑D kernel."""

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        *,
        shots: int = 100,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.n_qubits = kernel_size ** 2

        # Device
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)

        # Parameters for the variational part
        self.params = np.random.randn(self.n_qubits)

        # Build circuit
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor):
            # Data encoding: RX based on threshold
            for i in range(self.n_qubits):
                angle = np.pi if x[i] > self.threshold else 0.0
                qml.RX(angle, wires=i)
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=(i, i + 1))
            # Parameterised rotations
            for i in range(self.n_qubits):
                qml.RY(self.params[i], wires=i)
            # Expectation of Z on all wires
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, data):
        """Run the quantum circuit on classical data.

        Args:
            data: 2D array with shape (kernel_size, kernel_size).

        Returns:
            float: average expectation value of PauliZ over all qubits.
        """
        data = np.asarray(data).astype(np.float32)
        data = data.reshape(-1)  # flatten
        x = torch.tensor(data, dtype=torch.float32, device=self.circuit.device)
        out = self.circuit(x)  # list of tensors
        out = torch.stack(out, dim=0)
        return out.mean().item()

    def __call__(self, data):
        return self.run(data)


def Conv():
    """Convenience constructor matching the original API."""
    return ConvEnhanced(kernel_size=2, threshold=0.0, shots=100)


__all__ = ["ConvEnhanced", "Conv"]
