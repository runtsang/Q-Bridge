"""
Quantum self‑attention via a Pennylane variational circuit.

This module introduces a multi‑head quantum attention block:
  • Each head is mapped to a contiguous set of qubits.
  • Parameterised single‑qubit rotations encode query and key weights.
  • Controlled rotations entangle neighbouring qubits, modelling the
    key‑value interaction.
  • A measurement on each qubit yields a probability distribution that
    is interpreted as the attention score for that head.
  • The outputs of all heads are averaged to produce the final
    attention representation.

The public API mirrors the classical version: `run(backend, rotation_params,
entangle_params, shots=1024)`.

Typical usage:

    from SelfAttention import SelfAttention
    dev = qml.device("default.qubit", wires=16)
    attention = SelfAttention(n_qubits_per_head=4, num_heads=4)
    out = attention.run(dev, q_w, k_w, shots=1024)
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class SelfAttention:
    """Variational multi‑head quantum attention using Pennylane."""

    def __init__(self, n_qubits_per_head: int, num_heads: int):
        """
        Parameters
        ----------
        n_qubits_per_head : int
            Qubits allocated per attention head.
        num_heads : int
            Number of attention heads.
        """
        self.n_qubits_per_head = n_qubits_per_head
        self.num_heads = num_heads
        self.total_qubits = n_qubits_per_head * num_heads

        # Build a QNode
        @qml.qnode(qml.device("default.qubit", wires=self.total_qubits))
        def circuit(rotation_params, entangle_params, shots=None):
            # Apply parameterised rotations per qubit
            for h in range(self.num_heads):
                base = h * self.n_qubits_per_head
                for i in range(self.n_qubits_per_head):
                    idx = base + i
                    q_rot = rotation_params[h, i]
                    k_rot = entangle_params[h, i]
                    # Query encoding
                    qml.RX(q_rot[0], wires=idx)
                    qml.RY(q_rot[1], wires=idx)
                    qml.RZ(q_rot[2], wires=idx)
                    # Key encoding via controlled rotation with neighbour
                    if i < self.n_qubits_per_head - 1:
                        tgt = idx + 1
                        qml.CRX(k_rot, wires=[idx, tgt])
            # Measurement: return probabilities of all computational basis states
            return qml.probs(wires=range(self.total_qubits))

        self.circuit = circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the quantum self‑attention circuit.

        Parameters
        ----------
        backend : pennylane.Device
            Pennylane device (e.g., quantum simulator or real hardware).
        rotation_params : np.ndarray
            Shape (num_heads, n_qubits_per_head, 3) – RX, RY, RZ angles per head.
        entangle_params : np.ndarray
            Shape (num_heads, n_qubits_per_head - 1) – CKX angles per head.
        shots : int, optional
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Averaged attention scores of shape (num_heads, n_qubits_per_head).
        """
        # Rebuild circuit with the provided backend
        dev = backend
        @qml.qnode(dev)
        def circuit(rotation_params, entangle_params):
            for h in range(self.num_heads):
                base = h * self.n_qubits_per_head
                for i in range(self.n_qubits_per_head):
                    idx = base + i
                    q_rot = rotation_params[h, i]
                    qml.RX(q_rot[0], wires=idx)
                    qml.RY(q_rot[1], wires=idx)
                    qml.RZ(q_rot[2], wires=idx)
                    if i < self.n_qubits_per_head - 1:
                        tgt = idx + 1
                        k_rot = entangle_params[h, i]
                        qml.CRX(k_rot, wires=[idx, tgt])
            return qml.probs(wires=range(self.total_qubits))

        # Execute and collect probabilities
        probs = circuit(rotation_params, entangle_params)
        probs = probs.reshape(self.num_heads, self.n_qubits_per_head, -1)

        # For each head, sum probabilities for basis states grouped by qubit
        head_scores = np.zeros((self.num_heads, self.n_qubits_per_head))
        for h in range(self.num_heads):
            for i in range(self.n_qubits_per_head):
                # index of basis states where qubit i is |1>
                idx = 1 << (self.total_qubits - 1 - (h * self.n_qubits_per_head + i))
                head_scores[h, i] = probs[h, i, idx]

        # Average across heads to produce a single attention vector
        avg_attention = head_scores.mean(axis=0)  # shape (n_qubits_per_head,)
        return avg_attention

__all__ = ["SelfAttention"]
