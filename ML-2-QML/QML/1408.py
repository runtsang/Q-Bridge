"""Variational quantum circuit that emulates a fully connected layer.

The circuit uses Pennylane to build a parameterised ansatz consisting
of single‑qubit rotations followed by a layer of CNOT entangling
gates.  The ``run`` method accepts a list of rotation angles (``thetas``)
and returns the expectation value of a Pauli‑Z measurement on the first
qubit.  The implementation is compatible with both the local simulator
and a remote back‑end such as Qiskit Aer.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class FCL:
    """Parameterized quantum circuit with an entangling layer."""

    def __init__(self, n_qubits: int = 1, device_name: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device(device_name, wires=n_qubits, shots=1024)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: pnp.ndarray) -> pnp.ndarray:
            # Encoding layer
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            # Entangling layer
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measurement of Pauli‑Z on first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the circuit with the supplied rotation angles.

        Parameters
        ----------
        thetas : Iterable[float]
            Rotation angles for the RY gates.  The length must match
            ``n_qubits``; extra values are ignored.

        Returns
        -------
        np.ndarray
            Expectation value of the first qubit as a 1‑D array.
        """
        params = np.asarray(list(thetas)[: self.n_qubits], dtype=float)
        expval = self.circuit(params)
        return np.array([expval])


__all__ = ["FCL"]
