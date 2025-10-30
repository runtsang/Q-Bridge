"""Enhanced quantum convolutional filter using a variational ansatz.

This module defines :class:`ConvEnhanced`, a drop‑in replacement for the
original quantum filter.  It implements a parameter‑tuned variational
circuit with entanglement and a measurement‑based regularizer.  The
class exposes a ``run`` method that accepts a 2‑D array of shape
(k, k) and returns the average probability of measuring |1> across
qubits.
"""

import pennylane as qml
import numpy as np


class ConvEnhanced:
    """
    Variational quantum filter for 2‑D patches.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the filter (k × k).
    shots : int, default 200
        Number of shots for the circuit execution.
    threshold : float, default 0.5
        Threshold used to encode classical data into rotations.
    device_name : str, default "default.qubit"
        Pennylane device name.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 200,
        threshold: float = 0.5,
        device_name: str = "default.qubit",
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.dev = qml.device(device_name, wires=self.n_qubits, shots=shots)

        # Initialize variational parameters
        self.params = np.random.uniform(0, 2 * np.pi, size=(self.n_qubits,))

        @qml.qnode(self.dev)
        def circuit(inputs, params):
            # Data encoding
            for i in range(self.n_qubits):
                angle = np.pi if inputs[i] > self.threshold else 0.0
                qml.RX(angle, wires=i)

            # Variational rotations
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)

            # Entanglement layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Expectation of PauliZ for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, data):
        """
        Execute the quantum circuit on classical data.

        Parameters
        ----------
        data : array-like
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        # Flatten data to a 1‑D array matching the qubit ordering
        inputs = np.array(data).reshape(self.n_qubits)

        expvals = self.circuit(inputs, self.params)
        # Convert PauliZ expectation to probability of |1>
        probs = [(1 - e) / 2 for e in expvals]
        return float(np.mean(probs))


__all__ = ["ConvEnhanced"]
