"""Quantum convolution filter using Pennylane variational circuits.

The class implements a parameterised quantum circuit that maps a
2‑D kernel to a single expectation value.  It supports a simple
entanglement pattern and exposes a `gradient` method that returns the
parameter‑shift gradient with respect to the circuit weights.  The
`run` method keeps the original interface for compatibility.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class QuanvCircuit:
    """
    Variational quantum circuit for quanvolution layers.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel (maps to kernel_size**2 qubits).
    dev : pennylane.Device, optional
        Quantum device; defaults to the default.qubit simulator.
    cutoff : float, default 0.5
        Threshold for encoding classical data into rotation angles.
    """

    def __init__(self, kernel_size: int = 2, dev=None, cutoff: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.cutoff = cutoff
        self.dev = dev or qml.device("default.qubit", wires=self.n_qubits)
        # Initialise random weights for each qubit and each rotation
        self.weights = pnp.random.uniform(0, np.pi, (self.n_qubits, 3))
        self._circuit = self._build_ansatz()

    def _build_ansatz(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> float:
            # Initial Hadamard layer to create superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                # Encode classical data as RX rotation
                qml.RX(inputs[i], wires=i)
                # Parameterised rotations
                qml.RY(weights[i, 0], wires=i)
                qml.RZ(weights[i, 1], wires=i)
                qml.RX(weights[i, 2], wires=i)

            # Simple entangling pattern (nearest‑neighbour CNOTs)
            for i in range(0, self.n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])

            # Expectation value of PauliZ on the first qubit
            return qml.expval(qml.PauliZ(0))

        return circuit

    def run(self, data) -> float:
        """
        Execute the quantum circuit on a 2‑D kernel.

        Parameters
        ----------
        data : array‑like
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Expectation value of the circuit (in [-1, 1]).
        """
        arr = np.asarray(data, dtype=float).reshape(self.n_qubits)
        # Encode data as rotation angles: 0 if below cutoff, π otherwise
        inputs = np.pi * (arr > self.cutoff).astype(float)
        return float(self._circuit(inputs, self.weights))

    def gradient(self, data) -> np.ndarray:
        """
        Compute the parameter‑shift gradient of the circuit output
        with respect to the circuit weights.

        Parameters
        ----------
        data : array‑like
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        np.ndarray
            Gradient array with shape (n_qubits, 3).
        """
        arr = np.asarray(data, dtype=float).reshape(self.n_qubits)
        inputs = np.pi * (arr > self.cutoff).astype(float)
        return qml.grad(self._circuit)(inputs, self.weights)


def Conv(kernel_size: int = 2,
         dev=None,
         cutoff: float = 0.5) -> QuanvCircuit:
    """
    Factory that returns a configured QuanvCircuit instance.
    """
    return QuanvCircuit(kernel_size=kernel_size, dev=dev, cutoff=cutoff)


__all__ = ["Conv"]
