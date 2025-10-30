"""Quantum convolutional filter based on a variational circuit.

The filter accepts a 2‑D patch of data, encodes it into qubit rotations,
runs a depth‑controlled variational circuit, and returns the average
probability of measuring |1⟩ after post‑processing.  The circuit is
parameterised by a learnable rotation depth and an adaptive threshold,
allowing end‑to‑end optimisation with a classical optimiser.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class ConvEnhanced:
    """
    Variational quantum convolution filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the filter kernel.  The number of qubits is kernel_size².
    device : str, default "default.qubit"
        PennyLane backend to use.
    rotation_depth : int, default 2
        Number of variational layers.
    threshold : float, default 0.0
        Threshold applied to the per‑qubit probability of |1⟩.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        device: str = "default.qubit",
        rotation_depth: int = 2,
        threshold: float = 0.0,
    ) -> None:
        self.kernel_size = kernel_size
        self.num_qubits = kernel_size ** 2
        self.rotation_depth = rotation_depth
        self.threshold = threshold
        self.device = device
        self._setup_qnode()
        # Initialise parameters randomly
        self.params = pnp.random.uniform(
            0, 2 * np.pi, size=(self.rotation_depth, self.num_qubits)
        )

    def _setup_qnode(self) -> None:
        dev = qml.device(self.device, wires=self.num_qubits)

        @qml.qnode(dev, interface="autograd")
        def circuit(inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
            # Encode the classical data as RY rotations
            for i, val in enumerate(inputs):
                qml.RY(val * np.pi, wires=i)
            # Variational layers
            for depth in range(self.rotation_depth):
                for i in range(self.num_qubits):
                    qml.RY(params[depth, i], wires=i)
                # Ring‑type entanglement
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.num_qubits - 1, 0])
            # Return expectation values of Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.circuit = circuit

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum filter on a 2‑D patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1⟩ across all qubits after
            applying the adaptive threshold.
        """
        if data.ndim!= 2 or data.shape!= (self.kernel_size, self.kernel_size):
            raise ValueError(
                f"Input must be a 2‑D array of shape "
                f"({self.kernel_size}, {self.kernel_size})"
            )
        inputs = data.flatten()
        # Compute expectation values
        expvals = self.circuit(inputs, self.params)
        # Convert Z expectation values to probabilities of |1⟩
        probs = 0.5 * (1 - np.array(expvals))
        # Apply threshold
        probs = np.where(probs > self.threshold, 1.0, 0.0)
        return probs.mean()

    def set_params(self, params: np.ndarray) -> None:
        """
        Replace the internal variational parameters.

        Useful for optimisation loops that supply updated parameters.
        """
        if params.shape!= self.params.shape:
            raise ValueError("Parameter shape mismatch.")
        self.params = params

    def get_params(self) -> np.ndarray:
        """Return the current variational parameters."""
        return self.params


__all__ = ["ConvEnhanced"]
