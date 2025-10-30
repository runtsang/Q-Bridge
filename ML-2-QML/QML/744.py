"""Variational quantum circuit for a fully connected layer with expectation value."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Iterable

class FCL:
    """
    Variational quantum circuit that emulates a fully connected layer.
    The circuit consists of a layer of RX rotations followed by a
    controlled‑Z entangling pattern and a measurement of ⟨Z⟩ on the
    first qubit.  The parameters are treated as input features; the
    expectation value is returned as a numpy array.

    Parameters
    ----------
    n_qubits : int, default 3
        Number of qubits in the ansatz.
    depth : int, default 2
        Number of repeat layers.
    device : str, default "default.qubit"
        Pennylane device name.
    shots : int, default 1000
        Number of shots for the simulation.
    """
    def __init__(
        self,
        n_qubits: int = 3,
        depth: int = 2,
        device: str = "default.qubit",
        shots: int = 1000,
    ) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.device = qml.device(device, wires=n_qubits, shots=shots)
        self._build_circuit()

    def _build_circuit(self) -> None:
        @qml.qnode(self.device, interface="autograd")
        def circuit(params: pnp.ndarray) -> float:
            # Encode parameters as RX rotations
            for i, theta in enumerate(params):
                qml.RX(theta, wires=i % self.n_qubits)

            # Entangling pattern
            for _ in range(self.depth):
                for i in range(self.n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
                qml.CZ(wires=[0, self.n_qubits - 1])

            # Measurement of Pauli‑Z on the first qubit
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the variational circuit with the supplied parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameter vector.  Length must match ``n_qubits``; if shorter
            it is padded with zeros, if longer it is truncated.

        Returns
        -------
        np.ndarray
            Expectation value of the first qubit as a one‑element array.
        """
        params = np.array(thetas, dtype=float)
        if len(params) < self.n_qubits:
            params = np.pad(params, (0, self.n_qubits - len(params)), "constant")
        elif len(params) > self.n_qubits:
            params = params[: self.n_qubits]

        expectation = self._circuit(params)
        return np.array([expectation])

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the expectation value w.r.t. the parameters
        using the parameter‑shift rule.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameter vector.

        Returns
        -------
        np.ndarray
            Gradient vector of shape (n_qubits,).
        """
        params = np.array(thetas, dtype=float)
        if len(params) < self.n_qubits:
            params = np.pad(params, (0, self.n_qubits - len(params)), "constant")
        elif len(params) > self.n_qubits:
            params = params[: self.n_qubits]

        grad = qml.grad(self._circuit)(params)
        return grad

__all__ = ["FCL"]
