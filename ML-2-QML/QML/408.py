"""Variational quantum fully‑connected layer using PennyLane."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class FCL:
    """
    A quantum neural‑network layer that maps a list of theta parameters to
    expectation values of a Pauli‑Z observable on each wire.

    Parameters
    ----------
    n_qubits : int
        Number of qubits and input dimensionality.
    device : str, default 'default.qubit'
        PennyLane device identifier.
    wires : Sequence[int] | None, default None
        Wire indices; if None, defaults to range(n_qubits).
    n_entangling_layers : int, default 1
        Number of entanglement layers in the ansatz.
    """

    def __init__(
        self,
        n_qubits: int,
        device: str = "default.qubit",
        wires: Sequence[int] | None = None,
        n_entangling_layers: int = 1,
    ) -> None:
        self.n_qubits = n_qubits
        self.wires = wires or list(range(n_qubits))
        self.n_entangling_layers = n_entangling_layers
        self.dev = qml.device(device, wires=self.wires)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: Sequence[float]) -> Sequence[float]:
            # Encode parameters into Ry rotations
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=self.wires[i])

            # Entangling layers
            for _ in range(self.n_entangling_layers):
                for i in range(self.n_qubits - 1):
                    qml.CNOT(self.wires[i], self.wires[i + 1])
                for i in range(self.n_qubits):
                    qml.RZ(params[i], wires=self.wires[i])

            # Measure expectation of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        self._circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the ansatz for a single set of theta parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Input parameters (length must equal n_qubits).

        Returns
        -------
        np.ndarray
            Expectation values of shape (n_qubits,).
        """
        params = np.array(list(thetas), dtype=np.float64)
        return np.array(self._circuit(params))

    def gradients(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the expectation values with respect to
        each input parameter using PennyLane's autograd.

        Parameters
        ----------
        thetas : Iterable[float]
            Input parameters.

        Returns
        -------
        np.ndarray
            Gradient matrix of shape (n_qubits, n_qubits).
        """
        params = np.array(list(thetas), dtype=np.float64)
        return np.array(qml.grad(self._circuit)(params))

__all__ = ["FCL"]
