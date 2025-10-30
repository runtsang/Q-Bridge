"""Variational quantum self‑attention using Pennylane."""
from __future__ import annotations

import numpy as np
import pennylane as qml

class SelfAttentionModule:
    """
    Quantum self‑attention block implemented as a variational circuit.
    Parameters
    ----------
    n_qubits : int
        Number of qubits, typically equal to the embedding dimension.
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        @qml.qnode(self.dev)
        def circuit():
            # Apply parametrized single‑qubit rotations
            for i in range(self.n_qubits):
                qml.RY(rotation_params[3 * i], wires=i)
                qml.RZ(rotation_params[3 * i + 1], wires=i)
                qml.RX(rotation_params[3 * i + 2], wires=i)
            # Entangle adjacent qubits with controlled‑RX gates
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])
            # Return expectation values of Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Execute the quantum self‑attention circuit.
        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles for single‑qubit gates (length 3*n_qubits).
        entangle_params : np.ndarray
            Angles for controlled‑RX gates (length n_qubits-1).
        shots : int, optional
            Number of measurement shots; used only if a simulator supporting shots is chosen.
        Returns
        -------
        np.ndarray
            Expectation values of Pauli‑Z on each qubit, shape (n_qubits,).
        """
        circuit = self._circuit(rotation_params, entangle_params)
        # For the default.qubit simulator, shots are ignored; they are kept for API compatibility.
        return np.array(circuit())

__all__ = ["SelfAttentionModule"]
