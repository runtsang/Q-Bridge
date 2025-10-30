"""Quantum self‑attention implementation using Pennylane.

The module defines a ``QuantumAttention`` class that can be used as a drop‑in
replacement for the classical attention weight computation in the hybrid
``SelfAttention`` module.  The circuit is a simple variational ansatz that
produces a probability distribution over the sequence positions.  The
distribution is interpreted as attention weights.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Optional

class QuantumAttention:
    """Variational quantum circuit that outputs attention weights."""

    def __init__(self, *, embed_dim: int, weight_share: bool = False):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the input embeddings.  This value is not used
            directly by the quantum circuit but is kept for API compatibility.
        weight_share : bool, default=False
            If True, the rotation angles of the circuit are tied to the
            linear projection weights of the classical layer (not used in
            this simplified implementation).
        """
        self.embed_dim = embed_dim
        self.weight_share = weight_share
        # Number of qubits to encode a sequence of length L
        self.n_qubits = 4  # fixed for simplicity

        # Device for state‑vector simulation
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

    def _circuit(self, params: np.ndarray) -> np.ndarray:
        """Variational circuit that returns a probability distribution."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            # Encode parameters into RX rotations
            for i in range(self.n_qubits):
                qml.RX(params[i], wires=i)
            # Entangle
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure in computational basis
            probs = qml.probs(wires=range(self.n_qubits))
            return probs

        return circuit()

    def compute(self, scores: np.ndarray) -> np.ndarray:
        """
        Compute attention weights from a 2‑D array of scores.

        Parameters
        ----------
        scores : np.ndarray
            Shape (batch, seq_len).  Each row contains the raw attention
            scores for one example.

        Returns
        -------
        np.ndarray
            Shape (batch, seq_len, seq_len).  The outermost dimension
            indexes the batch, the middle dimension the query position,
            and the innermost dimension the key position.  The values
            represent the attention probability of each key for a given
            query.
        """
        batch, seq_len = scores.shape
        # Pad or truncate to 2^n_qubits
        target_len = 2 ** self.n_qubits
        if seq_len > target_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"{target_len} supported by the fixed 4‑qubit ansatz."
            )
        # Pad scores to target_len
        padded = np.pad(scores, ((0, 0), (0, target_len - seq_len)), mode="constant")
        # Normalise scores to angles in [0, pi]
        angles = np.clip(padded / np.max(np.abs(padded)), -1, 1) * np.pi
        # Compute probabilities for each batch element
        probs = np.array([self._circuit(angles[i]) for i in range(batch)])
        # Reshape to (batch, target_len)
        probs = probs.reshape(batch, target_len)
        # Truncate to original seq_len
        probs = probs[:, :seq_len]
        # For each query position, replicate the same distribution
        # (simple heuristic: same weights for all queries)
        weights = np.stack([probs] * seq_len, axis=1)
        return weights

# Alias to keep the original interface
SelfAttention = QuantumAttention
