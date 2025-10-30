"""Quantum implementation of a convolution‑style filter using Pennylane."""

import pennylane as qml
import numpy as np


class ConvFilterQuantum:
    """
    A parameterised quantum circuit that mimics a convolutional filter.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the square filter (default 2). The number of qubits is kernel_size**2.
    threshold : float, optional
        Data threshold to decide rotation angles (default 0.5).
    shots : int, optional
        Number of measurement shots (default 100).
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.5, shots: int = 100) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=shots)

        # The QNode will be built with an autograd interface for gradient support
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    # --------------------------------------------------------------------- #
    # Quantum circuit
    # --------------------------------------------------------------------- #
    def _circuit(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Build a simple entangled circuit with data‑dependent RX rotations
        followed by a trainable RY layer and a CNOT chain.
        """
        # Data encoding: RX rotations conditioned on data threshold
        for i in range(self.n_qubits):
            angle = np.pi if data[i] > self.threshold else 0.0
            qml.RX(angle, wires=i)

        # Parameterised rotation layer
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)

        # Entangling layer
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        # Expectation values of PauliZ on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def run(self, data: np.ndarray) -> float:
        """
        Evaluate the quantum filter on a 2D input array.

        Returns
        -------
        float
            Average absolute expectation value across all qubits, interpreted as a filter response.
        """
        # Flatten the data to match the qubit count
        flat = data.reshape(-1)
        if flat.size!= self.n_qubits:
            raise ValueError(f"Input size {flat.size} does not match expected {self.n_qubits} qubits.")

        # Random initial parameters; in practice these would be learned.
        params = np.random.uniform(0, 2 * np.pi, self.n_qubits)

        # Compute expectation values
        expvals = self.qnode(params, flat)

        # Return the mean absolute value as a simple scalar response
        return float(np.mean(np.abs(expvals)))


__all__ = ["ConvFilterQuantum"]
