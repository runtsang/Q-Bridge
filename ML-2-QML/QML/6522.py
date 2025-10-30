"""Quantum convolutional filter using Pennylane.

The circuit encodes a 2D patch of size `kernel_size`×`kernel_size`
into a register of `kernel_size**2` qubits.  Each qubit receives an
`RX` gate that depends on the pixel value, followed by a variational
layer of `RY` rotations and a ladder of `CNOT` gates.  The expectation
value of `Z` on the first qubit is returned as the filter response.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

__all__ = ["Conv"]

class QuanvCircuit:
    """Quantum convolutional filter."""

    def __init__(
        self,
        kernel_size: int = 2,
        device: qml.Device | None = None,
        shots: int | None = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots

        self.dev = (
            device
            if device is not None
            else qml.device("default.qubit", wires=self.n_qubits, shots=shots)
        )

        @qml.qnode(self.dev, interface="autograd")
        def circuit(data):
            # Data re‑uploading: RX gates encode pixel values
            for i in range(self.n_qubits):
                qml.RX(data[i], wires=i)
            # Variational layer: random RY rotations
            for i in range(self.n_qubits):
                qml.RY(np.random.uniform(0, 2 * np.pi), wires=i)
            # Entanglement ladder
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Expectation value of Z on the first qubit
            return qml.expval(qml.PauliZ(wires=0))

        self._circuit = circuit

    def run(self, data) -> float:
        """Return the expectation value for a 2D patch."""
        arr = np.asarray(data, dtype=np.float32)
        flat = arr.reshape(-1)
        # Map pixel values to [0, π] for RX rotation
        flat = np.clip(flat, 0, 1) * np.pi
        return float(self._circuit(flat))

def Conv(kernel_size: int = 2, shots: int | None = None) -> QuanvCircuit:
    """Return a quantum filter instance."""
    return QuanvCircuit(kernel_size, shots=shots)
