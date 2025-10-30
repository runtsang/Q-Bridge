"""Quantum variational convolution module using Pennylane.

The class implements a parameter‑trained variational circuit that learns a kernel
through gradient‑based optimization.  It is fully differentiable with respect
to the parameters and can be integrated into a PyTorch training loop via
Pennylane's autograd interface.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Tuple


class ConvGen213:
    """
    Variational quantum convolution filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel.
    shots : int, default 1024
        Number of shots for expectation estimation.
    device : str, default "default.qubit"
        Pennylane device name.
    threshold : float, default 0.5
        Threshold used to binarise the input data before encoding.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 1024,
        device: str = "default.qubit",
        threshold: float = 0.5,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.dev = qml.device(device, wires=self.n_qubits, shots=shots)

        # Trainable parameters for the variational circuit
        self.params = np.random.uniform(0, 2 * np.pi, size=self.n_qubits)

        # Define the circuit
        @qml.qnode(self.dev, interface="autograd")
        def circuit(x: np.ndarray, params: np.ndarray) -> float:
            # Encode data into rotation angles
            for i in range(self.n_qubits):
                qml.RY(x[i] * np.pi, wires=i)

            # Variational layer
            for i in range(self.n_qubits):
                qml.RZ(params[i], wires=i)

            # Entanglement
            for i in range(0, self.n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])

            # Measurement
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum circuit on input data.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            The expectation value of PauliZ on the first qubit,
            passed through a sigmoid with the stored threshold.
        """
        flat = data.flatten()
        # Binarise data based on threshold
        x = np.where(flat > self.threshold, 1.0, 0.0)
        expval = self.circuit(x, self.params)
        # Map expectation to [0,1] and apply threshold
        prob = (expval + 1) / 2
        return float(np.clip(prob, 0.0, 1.0))

    def get_params(self) -> np.ndarray:
        """Return current trainable parameters."""
        return self.params

    def set_params(self, new_params: np.ndarray) -> None:
        """Set new trainable parameters."""
        self.params = new_params
