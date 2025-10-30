"""Quantum convolutional filter.

Drop‑in replacement for the original Conv filter.
Implements a parameterised variational circuit that encodes the
input pixel values into RY rotations and measures the mean Z.
"""

import pennylane as qml
import pennylane.numpy as np
from typing import Iterable

class ConvCircuit:
    def __init__(self, kernel_size: int = 2, shots: int = 1024, threshold: float = 0.5):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        # parameters for the ansatz
        self.params = np.random.uniform(0, 2*np.pi, self.n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(*params, data):
            # data encoding
            for i, val in enumerate(data):
                theta = np.pi if val > self.threshold else 0.0
                qml.RY(theta, wires=i)
            # ansatz layers
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def run(self, data):
        """
        Execute the variational filter on a 2‑D array.

        Parameters
        ----------
        data : array‑like
            Shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Expectation value of Pauli‑Z on the first qubit.
        """
        data = np.array(data).flatten()
        return float(self._circuit(*self.params, data=data))

__all__ = ["ConvCircuit"]
