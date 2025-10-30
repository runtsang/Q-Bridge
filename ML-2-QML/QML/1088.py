"""
Quantum implementation of a fully‑connected layer using a variational circuit.
The circuit operates on ``n_qubits`` and returns the expectation value of
Pauli‑Z on the last qubit.  A parameter‑shift rule is provided for gradient
estimation, enabling integration into hybrid training loops.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class FCL:
    """
    Variational quantum circuit mimicking a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    dev : str or qml.Device, optional
        Quantum device to execute the circuit on.  If ``None``, a default
        Aer simulator is used.
    """

    def __init__(self, n_qubits: int = 1, dev: str | qml.Device | None = None) -> None:
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)

        # create a parameterised circuit
        self._theta = qml.numpy.array([0.0] * n_qubits, requires_grad=True)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: Sequence[float]) -> float:
            # initial rotation
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            # entanglement pattern
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # measure expectation of Z on the last qubit
            return qml.expval(qml.PauliZ(wires=n_qubits - 1))

        self._circuit = circuit

    def update_params(self, thetas: Sequence[float]) -> None:
        """
        Load a flat list of rotation angles into the circuit.

        Parameters
        ----------
        thetas : Sequence[float]
            List of angles, one per qubit.  If the length does not match,
            an exception is raised.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} parameters, got {len(thetas)}."
            )
        self._theta = pnp.array(thetas, requires_grad=True)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Flat list of rotation angles for the Ry gates.

        Returns
        -------
        np.ndarray
            Expectation value of Pauli‑Z on the last qubit, wrapped in a
            one‑element array for API consistency.
        """
        self.update_params(thetas)
        expval = self._circuit(self._theta)
        return np.array([expval])

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the expectation value w.r.t. the parameters
        using the parameter‑shift rule.

        Parameters
        ----------
        thetas : Iterable[float]
            Flat list of rotation angles.

        Returns
        -------
        np.ndarray
            Gradient vector of shape (n_qubits,).
        """
        self.update_params(thetas)
        grad_fn = qml.gradients.param_shift(self._circuit)
        grad = grad_fn(self._theta)
        return np.array(grad)

__all__ = ["FCL"]
