"""
Quantum convolutional filter implemented with PennyLane.
Provides a parameterized circuit that can be trained via backpropagation.
Designed to be drop-in compatible with the original Conv() function.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from typing import Tuple, Union

class ConvGen316:
    """
    Quantum convolutional filter using PennyLane.
    Parameters
    ----------
    kernel_size : int or Tuple[int, int]
        Size of the convolution kernel. If an int, the kernel is square.
    threshold : float, default 0.0
        Threshold applied to input data before encoding.
    dev_name : str, default "default.qubit"
        PennyLane device name.
    shots : int, default 1000
        Number of shots for probability estimation.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 2,
        threshold: float = 0.0,
        dev_name: str = "default.qubit",
        shots: int = 1000,
    ) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size[0] * kernel_size[1]
        self.dev = qml.device(dev_name, wires=self.n_qubits, shots=shots)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, params):
            # Encode data via RX rotations
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
            # Parameterized layer
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            # Simple entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.probs(wires=range(self.n_qubits))

        self.circuit = circuit
        # Initialize trainable parameters
        self.params = np.random.uniform(0, 2 * np.pi, self.n_qubits)

    def run(self, data: np.ndarray) -> float:
        """
        Evaluate the quantum filter on a single 2D array.
        Parameters
        ----------
        data : np.ndarray
            2D array of shape (H, W).
        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        flattened = data.flatten()
        # Apply threshold encoding
        encoded = np.where(flattened > self.threshold, np.pi, 0.0)
        probs = self.circuit(encoded, self.params)
        probs = np.array(probs)
        # Compute marginal probability for each qubit
        marginal = np.zeros(self.n_qubits)
        for i in range(self.n_qubits):
            mask = ((np.arange(2 ** self.n_qubits) >> i) & 1) == 1
            marginal[i] = probs[mask].sum()
        return marginal.mean()

    def get_params(self):
        """Return current trainable parameters."""
        return self.params

    def set_params(self, new_params):
        """Set trainable parameters."""
        self.params = new_params

def Conv(kernel_size: Union[int, Tuple[int, int]] = 2,
         threshold: float = 0.0,
         dev_name: str = "default.qubit",
         shots: int = 1000) -> ConvGen316:
    """Return a ConvGen316 instance."""
    return ConvGen316(kernel_size, threshold, dev_name, shots)

__all__ = ["ConvGen316", "Conv"]
