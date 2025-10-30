"""Quantum variational fully‑connected layer implemented with PennyLane.

The circuit uses a parameterized rotation layer followed by a measurement of the
Pauli‑Z expectation value.  The class supports automatic differentiation via
PennyLane, making it suitable for hybrid classical‑quantum training.

Features
--------
- Multi‑qubit support (default 2 qubits for a richer representation).
- Parameterized rotation angles with a simple RY layer.
- `run` method returning a NumPy array of the expectation value.
- `state` property exposing the current quantum state for diagnostics.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class FCL:
    """
    Variational quantum circuit acting as a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    dev_name : str
        PennyLane device name (e.g. ``default.qubit``).
    """

    def __init__(self, n_qubits: int = 2, dev_name: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(thetas: List[float]):
            # Parameterized rotation layer
            for i in range(n_qubits):
                qml.RY(thetas[i], wires=i)
            # Simple entangling layer
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure expectation of Z on the first qubit
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit and return the expectation value.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of rotation angles, one per qubit.

        Returns
        -------
        np.ndarray
            Array containing the expectation value of Pauli‑Z on qubit 0.
        """
        theta_list = list(thetas)
        # Pad or truncate to match the number of qubits
        if len(theta_list) < self.n_qubits:
            theta_list += [0.0] * (self.n_qubits - len(theta_list))
        elif len(theta_list) > self.n_qubits:
            theta_list = theta_list[: self.n_qubits]

        expval = self._circuit(theta_list)
        return np.array([expval])

    @property
    def state(self) -> np.ndarray:
        """Return the current statevector of the device."""
        return self.dev.state

__all__ = ["FCL"]
