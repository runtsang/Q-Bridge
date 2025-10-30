"""
A quantum implementation of a fully‑connected layer that maps a scalar input
to an expectation value via a parameterised variational circuit.

The public API mirrors the classical version: ``run(thetas)`` returns a
one‑element NumPy array.  The circuit consists of a chain of rotation
gates followed by a full‑SWAP entangling network, and the observable is Z
on the last qubit.  The backend, number of qubits and shots are
configurable.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Iterable, Sequence, Optional


class FullyConnectedLayer:
    """
    Variational fully‑connected layer implemented with Pennylane.

    Parameters
    ----------
    n_qubits:
        Number of qubits used in the circuit.
    dev:
        Pennylane quantum device.  Defaults to ``default.qubit``.
    shots:
        Number of shots for state‑vector or measurement execution.  ``None``
        triggers analytic expectation evaluation.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        dev: Optional[qml.Device] = None,
        shots: Optional[int] = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = dev or qml.device("default.qubit", wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: np.ndarray) -> np.ndarray:
            qml.RY(params[0], wires=0)
            for i in range(n_qubits):
                qml.RZ(params[1 + i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[i + 1, i])
            return qml.expval(qml.PauliZ(self.n_qubits - 1))

        self._circuit = circuit

    # ---------------------------------------------------------------------------

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the variational circuit with the supplied parameters.

        Parameters
        ----------
        thetas:
            Iterable of rotation angles.  Length must be ``1 + n_qubits``.
        Returns
        -------
        np.ndarray
            One‑element array containing the expectation value of Z on the
            last qubit.
        """
        params = np.array(list(thetas), dtype=np.float64)
        if params.size!= 1 + self.n_qubits:
            raise ValueError(
                f"Expected {1 + self.n_qubits} parameters, got {params.size}"
            )
        expval = self._circuit(params)
        return np.array([float(expval)])

    # ---------------------------------------------------------------------------

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """Return the gradient of the expectation value w.r.t. the parameters.

        Uses the automatic differentiation backend of Pennylane.
        """
        params = np.array(list(thetas), dtype=np.float64)
        grad_fn = qml.grad(self._circuit)
        grads = grad_fn(params)
        return grads

    # ---------------------------------------------------------------------------

    def parameters_vector_length(self) -> int:
        """Return the expected number of parameters."""
        return 1 + self.n_qubits


def FCL(
    n_qubits: int = 1,
    dev: Optional[qml.Device] = None,
    shots: Optional[int] = None,
) -> FullyConnectedLayer:
    """Convenience factory mirroring the original API."""
    return FullyConnectedLayer(n_qubits=n_qubits, dev=dev, shots=shots)


__all__ = ["FullyConnectedLayer", "FCL"]
