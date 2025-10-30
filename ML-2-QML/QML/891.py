"""
Quantum fully‑connected layer implemented with a variational circuit.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class FCLayer:
    """
    Variational quantum circuit that emulates a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (features) in the circuit.
    shots : int, optional
        Number of measurement shots. If ``None`` the device runs in
        state‑vector mode.
    """

    def __init__(self, n_qubits: int = 1, shots: int | None = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        # Build the QNode; interface='autograd' allows us to compute gradients
        @qml.qnode(self.dev, interface="autograd")
        def circuit(thetas):
            # Parameterised single‑qubit rotations
            for i in range(n_qubits):
                qml.RY(thetas[i], wires=i)

            # Simple entangling layer
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Read‑out expectation of Pauli‑Z on the first qubit
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the variational circuit with the supplied parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of rotation angles; must match ``n_qubits`` in length.

        Returns
        -------
        np.ndarray
            Array of shape (1,) containing the expectation value of
            Pauli‑Z on the first qubit.
        """
        theta_array = pnp.array(list(thetas), dtype=pnp.float32)
        if theta_array.shape[0]!= self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} parameters, got {theta_array.shape[0]}"
            )
        expectation = self._circuit(theta_array)
        return np.array([expectation])

__all__ = ["FCLayer"]
