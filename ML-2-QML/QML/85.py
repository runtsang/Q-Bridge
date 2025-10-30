"""Quantum self‑attention using PennyLane.

This module implements a variational circuit that mirrors the
classical multi‑head attention block.  The circuit is parameterised
by rotation and entanglement angles and returns expectation values
that can be interpreted as attention weights.

The design is deliberately modular so it can be swapped into a
hybrid quantum‑classical training loop.
"""

import pennylane as qml
import numpy as np

class SelfAttentionModel:
    """
    Variational self‑attention circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (typically equal to the embedding dimension).
    num_heads : int, default 1
        Number of independent attention heads; each head uses its own
        slice of the qubit register.
    """
    def __init__(self, n_qubits: int, num_heads: int = 1):
        self.n_qubits = n_qubits
        self.num_heads = num_heads
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(rotation_params: np.ndarray, entangle_params: np.ndarray):
            """
            Parameters
            ----------
            rotation_params : np.ndarray
                Shape (n_qubits, 3) – RX, RY, RZ angles per qubit.
            entangle_params : np.ndarray
                Shape (n_qubits - 1,) – CRX angles between adjacent qubits.
            """
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(rotation_params[i, 0], wires=i)
                qml.RY(rotation_params[i, 1], wires=i)
                qml.RZ(rotation_params[i, 2], wires=i)

            # Entanglement layer
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Return expectation values of Pauli‑Z as attention logits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit and return expectation values.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles for each qubit.
        entangle_params : np.ndarray
            Entanglement angles between adjacent qubits.
        shots : int, default 1024
            Number of shots for the backend.  Ignored when using the
            default simulator but kept for API compatibility.

        Returns
        -------
        np.ndarray
            Array of expectation values of shape (n_qubits,).
        """
        # For the simulator the shot count is not used; for real devices
        # one would pass a backend that accepts shots.
        return np.array(self.circuit(rotation_params, entangle_params))

__all__ = ["SelfAttentionModel"]
