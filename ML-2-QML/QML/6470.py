"""Quantum convolutional filter using a variational circuit for quanvolution layers."""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane import qnode


class Conv:
    """
    Quantum convolutional filter that emulates the classical counterpart.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel; number of qubits = kernel_size ** 2.
    backend : str, default "default.qubit"
        PennyLane device backend.
    shots : int, default 1024
        Number of shots for expectation estimation.
    threshold : float, default 127.0
        Threshold used to decide rotation angles.
    depth : int, default 2
        Number of variational layers.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: str = "default.qubit",
        shots: int = 1024,
        threshold: float = 127.0,
        depth: int = 2,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.depth = depth

        self.dev = qml.device(backend, wires=self.n_qubits, shots=shots)

        # Define variational circuit
        def circuit(data):
            # Encode data into rotation angles
            for i, val in enumerate(data):
                angle = pnp.pi if val > self.threshold else 0.0
                qml.RX(angle, wires=i)

            # Variational layers
            for _ in range(self.depth):
                for i in range(self.n_qubits):
                    qml.RY(pnp.Variable("theta{}".format(i)), wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(self.n_qubits):
                    qml.RZ(pnp.Variable("theta{}".format(i)), wires=i)

            # Measure expectation of PauliZ on all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = qml.QNode(circuit, self.dev)

        # Initialize parameters
        self.params = np.random.uniform(0, 2 * np.pi, self.n_qubits)

    def run(self, data) -> float:
        """
        Execute the variational circuit on the provided data.

        Parameters
        ----------
        data : array‑like
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data_flat = np.reshape(data, (self.n_qubits,))
        # Bind parameters
        param_dict = {f"theta{i}": self.params[i] for i in range(self.n_qubits)}
        expectation = self.qnode(data_flat, **param_dict)

        # Convert expectation values to probabilities of |1>
        probs = [(1 - e) / 2 for e in expectation]
        return np.mean(probs).item()


__all__ = ["Conv"]
