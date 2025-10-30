"""Variational quantum self‑attention using Pennylane.

The quantum circuit emulates a single attention head.  Parameters
(`rotation_params` and `entangle_params`) are applied as rotations and
CZ‑entanglements on each qubit.  The output is a probability vector
over the qubits that can be interpreted as attention weights.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class SelfAttention:
    """Variational quantum self‑attention.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (one per token in the sequence).
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=1024)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
        ) -> np.ndarray:
            """Apply a parameterised circuit and return measurement probabilities."""
            # Encode the inputs as amplitude‑encoded state
            qml.QubitStateVector(inputs, wires=range(n_qubits))

            # Rotation layer
            for i in range(n_qubits):
                idx = 3 * i
                qml.RX(rotation_params[idx], wires=i)
                qml.RY(rotation_params[idx + 1], wires=i)
                qml.RZ(rotation_params[idx + 2], wires=i)

            # Entanglement layer
            for i in range(n_qubits - 1):
                qml.CZ(entangle_params[i], wires=[i, i + 1])

            # Measure all qubits in the computational basis
            return qml.probs(wires=range(n_qubits))

        self.circuit = circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Execute the circuit and return the probability distribution.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles (length 3 * n_qubits).
        entangle_params : np.ndarray
            Entanglement angles (length n_qubits-1).
        inputs : np.ndarray
            Amplitude‑encoded vector of shape (2**n_qubits,).

        Returns
        -------
        np.ndarray
            Probability vector of shape (2**n_qubits,).
        """
        probs = self.circuit(rotation_params, entangle_params, inputs)
        return probs.detach().numpy()

# expose the class for import
__all__ = ["SelfAttention"]
