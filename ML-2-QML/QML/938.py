"""Quantum self‑attention using Pennylane.

This module implements a self‑attention block entirely on a quantum device.
The circuit encodes the input embeddings into rotation angles and uses a
variational ansatz to generate a probability‑based attention map.
"""

import numpy as np
import pennylane as qml
import torch

class SelfAttention:
    """
    Quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits, typically equal to the sequence length.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """Build a variational circuit that outputs a probability distribution."""

        @qml.qnode(self.dev, interface="torch")
        def circuit():
            # Encode inputs as rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])
            # Measure all qubits in computational basis
            return [qml.probs(wires=i) for i in range(self.n_qubits)]

        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the quantum circuit and return a probability‑weighted attention map.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles for each qubit (shape (3 * n_qubits,)).
        entangle_params : np.ndarray
            Entangling angles (shape (n_qubits - 1,)).
        shots : int, default 1024
            Number of shots for sampling.

        Returns
        -------
        np.ndarray
            Attention weights of shape (n_qubits, n_qubits).
        """
        circuit = self._circuit(rotation_params, entangle_params)
        probs = circuit()
        # probs is a list of tensors for each qubit; convert to numpy
        probs_np = [p.detach().numpy() for p in probs]
        # Build a vector of probabilities of measuring |1> for each qubit
        weights = np.array([p[1] for p in probs_np])  # shape (n_qubits,)
        # Normalize to sum to one across sequence length
        attn_weights = weights / weights.sum()
        # Broadcast to square matrix
        return np.outer(attn_weights, attn_weights)
