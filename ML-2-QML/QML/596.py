# qml_code: FCL__gen245.py

"""
Quantum implementation of a fully‑connected layer using Pennylane.
The circuit is a shallow variational ansatz that maps a classical
parameter vector to an expectation value of a Pauli‑Z observable.

The API matches the original seed (`run(thetas: Iterable[float]) -> np.ndarray`)
while providing automatic differentiation and support for both
simulated and real backends.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class FCL:
    """
    Variational quantum circuit that emulates a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (equal to the length of the input vector).
    dev_name : str, optional
        Pennylane device name (default: "default.qubit").
    shots : int, optional
        Number of shots for measurement. If None, uses state‑vector
        simulation and returns exact expectation values.
    """

    def __init__(
        self,
        n_qubits: int,
        dev_name: str = "default.qubit",
        shots: int | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = qml.device(dev_name, wires=n_qubits, shots=shots)

        # Create a variational circuit template
        @qml.qnode(self.dev, interface="autograd", diff_method="backprop")
        def circuit(params):
            # Encode each input parameter with an RY rotation
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)

            # Entangling layer (CNOT ladder)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Measurement of Z on the first qubit
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit for the supplied parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            List of rotation angles, one per qubit.

        Returns
        -------
        np.ndarray
            Array containing the expectation value of the measurement.
        """
        params = pnp.array(list(thetas), dtype=pnp.float64)
        expectation = self._circuit(params)
        return np.array([expectation])

    def grad(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the expectation value w.r.t. the input parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            List of rotation angles.

        Returns
        -------
        np.ndarray
            Gradient vector of shape (n_qubits,).
        """
        params = pnp.array(list(thetas), dtype=pnp.float64)
        gradient = qml.grad(self._circuit)(params)
        return np.array(gradient)

__all__ = ["FCL"]
