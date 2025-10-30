"""Variational quantum self‑attention using Pennylane.

The circuit implements a parameterised rotation layer followed by a
controlled‑rotation entanglement pattern.  The measurement results
are interpreted as attention logits over the input qubits.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class QuantumSelfAttention:
    """
    Variational circuit that produces a probability distribution over qubits.
    Parameters:
        n_qubits (int): Number of qubits (also the sequence length).
        n_layers (int): Depth of the rotation‑entanglement stack.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _rotation_layer(self, params: np.ndarray):
        """Apply a full SU(2) rotation to every qubit."""
        for i in range(self.n_qubits):
            qml.Rot(params[3 * i], params[3 * i + 1], params[3 * i + 2], wires=i)

    def _entanglement_layer(self, params: np.ndarray):
        """
        Apply a chain of controlled‑RZ gates.
        params should have length n_qubits - 1.
        """
        for i in range(self.n_qubits - 1):
            qml.CRX(params[i], wires=[i, i + 1])

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        self._rotation_layer(rotation_params)
        self._entanglement_layer(entangle_params)

    @qml.qnode
    def _variational(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        self._circuit(rotation_params, entangle_params)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit and return a softmax‑normalised attention vector.
        """
        # Ensure parameter shapes
        if rotation_params.size!= 3 * self.n_qubits:
            raise ValueError("rotation_params must be of length 3 * n_qubits")
        if entangle_params.size!= self.n_qubits - 1:
            raise ValueError("entangle_params must be of length n_qubits - 1")

        # Expectation values are in [-1, 1]; shift to [0, 2] and softmax
        raw = self._variational(rotation_params, entangle_params)
        raw = np.array(raw)
        logits = (raw + 1) / 2
        return np.exp(logits) / np.sum(np.exp(logits))

__all__ = ["QuantumSelfAttention"]
