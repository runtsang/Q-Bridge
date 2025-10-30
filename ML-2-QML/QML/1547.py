"""Quantum self‑attention via a variational circuit implemented in PennyLane."""
from __future__ import annotations

import pennylane as qml
import numpy as np
import torch

class QuantumSelfAttentionModule:
    """
    Variational quantum circuit that mimics a self‑attention block.

    The circuit applies a parameterised rotation to each qubit followed by a chain of
    controlled‑X gates (CNOTs).  After the circuit we measure the expectation of
    Pauli‑Z on each qubit; these values are interpreted as attention scores
    which are combined with the input via a simple weighted sum.

    Parameters
    ----------
    n_qubits : int
        Number of qubits; typically set to the number of tokens in the sequence.
    seed : int | None
        Random seed for device initialization.
    """

    def __init__(self, n_qubits: int = 4, seed: int | None = None) -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=1024)
        self._build_circuit(seed)

    def _build_circuit(self, seed: int | None) -> None:
        """Create a parameter‑shaped circuit."""

        @qml.qnode(self.dev)
        def circuit(rotation_params: np.ndarray, entangle_params: np.ndarray):
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Entangling layer (CNOT chain)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                # Optional tunable controlled‑X strength
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the variational attention circuit and combine the expectation
        values with the input embeddings.

        Parameters
        ----------
        backend : qiskit.backends.AerSimulator | None
            Ignored in the PennyLane implementation but kept for API parity.
        rotation_params : np.ndarray
            Rotation angles for each qubit (shape ``(3*n_qubits,)``).
        entangle_params : np.ndarray
            Entangling gate strengths (shape ``(n_qubits-1,)``).
        inputs : np.ndarray
            Input embeddings of shape ``(batch_size, seq_len, embed_dim)``.
            The first ``seq_len`` tokens are mapped to qubits.
        shots : int, default 1024
            Number of measurement shots for expectation estimation.

        Returns
        -------
        np.ndarray
            Attention‑weighted representation of shape ``(batch_size, seq_len, embed_dim)``.
        """
        # Evaluate circuit to obtain expectation values
        exp_vals = self.circuit(rotation_params, entangle_params)

        # Convert to numpy array and normalize to [0,1]
        scores = (np.array(exp_vals) + 1) / 2.0  # Pauli‑Z -> [0,1]

        # Broadcast scores over batch and embed_dim
        scores = np.expand_dims(scores, axis=0)  # (1, seq_len)
        scores = np.repeat(scores, inputs.shape[0], axis=0)  # (batch_size, seq_len)

        # Weighted sum over sequence dimension
        weighted = np.einsum("bti,bi->bt", inputs, scores)

        # Expand to match input shape for compatibility
        output = np.tile(weighted[..., None], (1, 1, inputs.shape[2]))

        return output

__all__ = ["QuantumSelfAttentionModule"]
