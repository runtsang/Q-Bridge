"""
FullyConnectedLayer (FCL) – Quantum variational layer.

Features
--------
* Multi‑qubit entangled ansatz with alternating RX/RY rotations.
* Parameter vector interface compatible with the original seed.
* Expectation value of Pauli‑Z on the last qubit.
* Gradient via parameter‑shift rule (Pennylane).
"""

import pennylane as qml
import numpy as np
from typing import Iterable

class FCL:
    """
    Variational quantum circuit that mimics a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the ansatz.
    layers : int, optional
        Number of repeat layers. Defaults to 2.
    dev : pennylane.Device, optional
        Quantum device. If None, a default.qubit simulator is used.
    """
    def __init__(self, n_qubits: int = 1, layers: int = 2, dev: qml.Device = None) -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)
        self.n_params = n_qubits * layers * 2  # RX and RY per qubit per layer

        @qml.qnode(self.dev, interface="autograd")
        def circuit(theta):
            """Parameterized circuit."""
            theta = theta.reshape(self.layers, self.n_qubits, 2)
            for l in range(self.layers):
                for q in range(self.n_qubits):
                    qml.RX(theta[l, q, 0], wires=q)
                    qml.RY(theta[l, q, 1], wires=q)
                # Entangle all qubits with a chain of CNOTs
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            # Expectation of PauliZ on the last qubit
            return qml.expval(qml.PauliZ(self.n_qubits - 1))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit with a supplied parameter vector.

        Parameters
        ----------
        thetas : Iterable[float]
            Flat vector of rotation angles.

        Returns
        -------
        np.ndarray
            Expectation value as a 1‑D array.
        """
        theta = np.asarray(thetas, dtype=np.float32)
        assert theta.size == self.n_params, (
            f"Expected {self.n_params} parameters, got {theta.size}"
        )
        expval = self.circuit(theta)
        return np.array([expval])

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the expectation w.r.t. the parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Flat vector of rotation angles.

        Returns
        -------
        np.ndarray
            Gradient vector of the same length as `thetas`.
        """
        theta = np.asarray(thetas, dtype=np.float32)
        grad = qml.grad(self.circuit)(theta)
        return grad.reshape(-1)

__all__ = ["FCL"]
