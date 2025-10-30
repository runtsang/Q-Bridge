"""Enhanced quantum fully‑connected layer using a variational circuit.

The implementation replaces the simple H‑RY circuit with a multi‑qubit
parameterised circuit that includes entanglement and a Pauli‑Z
measurement.  The `run` method remains compatible with the original
API, and a gradient helper is provided for hybrid optimisation.
"""

from __future__ import annotations

from typing import Iterable
import numpy as np
import pennylane as qml
from pennylane import numpy as qnp


class EnhancedFCL:
    """
    Variational quantum circuit mimicking a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (default 1).
    dev : pennylane.Device, optional
        Quantum device; defaults to the default.qubit simulator.
    """

    def __init__(self, n_qubits: int = 1, dev: qml.Device | None = None) -> None:
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Define a parameterised circuit with local RY rotations and CNOT entanglement."""

        @qml.qnode(self.dev, interface="autograd")
        def circuit(thetas: qnp.ndarray) -> list[qnp.ndarray]:
            # Apply a local rotation to each qubit
            for i in range(self.n_qubits):
                qml.RY(thetas[i], wires=i)
            # Entangle neighbouring qubits
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Return expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit for a batch of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameter list; length must equal `n_qubits`.

        Returns
        -------
        np.ndarray
            Array of expectation values (one per qubit).
        """
        thetas_arr = qnp.array(list(thetas))
        if thetas_arr.shape[0]!= self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} parameters, got {thetas_arr.shape[0]}"
            )
        return np.array(self.circuit(thetas_arr))

    def grad(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the circuit outputs w.r.t. the parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameter list; length must equal `n_qubits`.

        Returns
        -------
        np.ndarray
            Gradient matrix of shape (n_qubits, n_qubits), where
            grad[i, j] = ∂⟨Z_j⟩ / ∂θ_i.
        """
        thetas_arr = qnp.array(list(thetas))
        return qml.grad(self.circuit)(thetas_arr)

__all__ = ["EnhancedFCL"]
