"""Quantum kernel construction using Pennylane variational ansatz."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp


class QuantumKernel:
    """
    Quantum kernel evaluated via a parameterised variational circuit.

    The circuit encodes two classical feature vectors x and y by applying
    rotation gates.  The kernel value is the expectation value of the PauliZ
    operator on a single qubit after the encoding and a simple entangling
    block.  The circuit can be extended with additional layers or different
    entangling patterns.

    Parameters
    ----------
    num_qubits : int, default=4
        Number of qubits used to encode the data.
    num_layers : int, default=2
        Number of entangling layers in the ansatz.
    """

    def __init__(self, num_qubits: int = 4, num_layers: int = 2) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(x: Sequence[float], y: Sequence[float]) -> pnp.ndarray:
            # Feature encoding
            for i in range(self.num_qubits):
                qml.RY(x[i], wires=i)

            # Entangling layers
            for _ in range(self.num_layers):
                for i in range(self.num_qubits):
                    qml.RZ(np.pi / 4, wires=i)
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # Reverse encoding of y
            for i in range(self.num_qubits):
                qml.RY(-y[i], wires=i)

            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, X: Sequence[Sequence[float]], Y: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of feature vectors.

        Parameters
        ----------
        X : Sequence[Sequence[float]]
            First dataset of shape (n_samples_X, n_features).
        Y : Sequence[Sequence[float]]
            Second dataset of shape (n_samples_Y, n_features).

        Returns
        -------
        numpy.ndarray
            Kernel matrix of shape (len(X), len(Y)).
        """
        K = np.zeros((len(X), len(Y)))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i, j] = self.circuit(x, y)
        return K

    def kernel_matrix(self, a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Legacy helper that mimics the original API.

        Parameters
        ----------
        a : Sequence[Sequence[float]]
        b : Sequence[Sequence[float]]

        Returns
        -------
        numpy.ndarray
            Pairwise kernel matrix.
        """
        return self.forward(a, b)


def kernel_matrix(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Convenience wrapper to keep the original function signature.

    Parameters
    ----------
    a : Sequence[Sequence[float]]
    b : Sequence[Sequence[float]]

    Returns
    -------
    numpy.ndarray
        Pairwise kernel matrix.
    """
    kernel = QuantumKernel()
    return kernel.forward(a, b)


__all__ = ["QuantumKernel", "kernel_matrix"]
