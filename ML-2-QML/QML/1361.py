"""Quantum self‑attention with Pennylane.

The implementation follows the seed's interface but replaces the
bare Qiskit circuit with a variational circuit that can be trained
by gradient‑based optimisers.  The circuit uses rotation parameters
to create a query‑like state and entanglement parameters to
implement a key‑like interaction.  The resulting measurement
expectations are interpreted as the attention weights.
"""

import pennylane as qml
import numpy as np
from typing import Tuple

class QuantumSelfAttention:
    """
    Variational self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be at least 2).
    num_layers : int, default 1
        Number of variational layers.
    seed : int, optional
        Random seed for weight initialization.
    """

    def __init__(self, n_qubits: int, num_layers: int = 1, seed: int = 42) -> None:
        if n_qubits < 2:
            raise ValueError("n_qubits must be at least 2")
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._build_variational()

    def _build_variational(self) -> None:
        """Create a parameterised circuit that accepts rotation and entanglement arrays."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit(rotation_params: np.ndarray,
                    entangle_params: np.ndarray) -> np.ndarray:
            # Rotation block: each qubit receives a 3‑parameter Ry‑Rz‑Ry sequence
            for i in range(self.n_qubits):
                qml.RY(rotation_params[3 * i], wires=i)
                qml.RZ(rotation_params[3 * i + 1], wires=i)
                qml.RY(rotation_params[3 * i + 2], wires=i)

            # Entanglement block: a chain of controlled‑RZ gates
            for i in range(self.n_qubits - 1):
                qml.CZ(wires=[i, i + 1])
                qml.RZ(entangle_params[i], wires=i + 1)

            # Measurement: expectation of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        self._circuit = circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the variational circuit and return the expectation values.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of shape ``(3 * n_qubits,)`` containing rotation angles.
        entangle_params : np.ndarray
            Array of shape ``(n_qubits - 1,)`` containing entanglement angles.
        shots : int, default 1024
            Number of shots for the backend simulator.  When using the
            default autograd device the shots argument is ignored.

        Returns
        -------
        np.ndarray
            1‑D array of length ``n_qubits`` containing the expectation
            values of Pauli‑Z.  These values are interpreted as the
            attention weights (after normalisation).
        """
        # The default autograd device returns expectation values directly.
        # For a real quantum backend we would use qiskit or a cloud provider.
        raw = self._circuit(rotation_params, entangle_params)
        # Normalise to a probability distribution
        attn = raw - raw.min()
        attn = attn / attn.sum()
        return np.asarray(attn)

def SelfAttention() -> QuantumSelfAttention:
    """Factory mirroring the seed interface."""
    return QuantumSelfAttention(n_qubits=4, num_layers=1)

__all__ = ["SelfAttention"]
