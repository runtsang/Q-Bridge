"""
Variational quantum self‑attention using PennyLane.

The circuit encodes classical tokens via rotations, applies
parameter‑dependent single‑qubit rotations and entangling gates,
and measures expectation values of Pauli‑Z to form attention
weights.  The interface matches the classical version so it can be
swapped in experiments.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Tuple

class QuantumSelfAttention:
    """
    Quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (one per token position).
    num_heads : int, default 1
        Number of independent attention heads; each head uses a
        separate device instance.
    """

    def __init__(self, n_qubits: int, num_heads: int = 1):
        self.n_qubits = n_qubits
        self.num_heads = num_heads
        self.devices = [
            qml.device("default.qubit", wires=n_qubits) for _ in range(num_heads)
        ]

    def _circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a variational circuit that maps input embeddings to
        expectation values used as attention scores.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (n_qubits, 3) – RX, RY, RZ angles per qubit.
        entangle_params : np.ndarray
            Shape (n_qubits - 1,) – CX phase angles between neighbours.
        inputs : np.ndarray
            Shape (n_qubits,) – classical token embeddings to encode.

        Returns
        -------
        scores : np.ndarray
            Normalized attention weights (softmax over qubit expectations).
        """
        @qml.qnode(self.devices[0])
        def circuit():
            # Encode classical data via rotations
            for i in range(self.n_qubits):
                qml.RX(inputs[i] * np.pi, wires=i)
            # Parameterized rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[i, 0], wires=i)
                qml.RY(rotation_params[i, 1], wires=i)
                qml.RZ(rotation_params[i, 2], wires=i)
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        raw_expectations = circuit()
        # Convert to probabilities
        logits = np.array(raw_expectations)
        logits = (logits + 1) / 2  # shift from [-1,1] to [0,1]
        scores = logits / logits.sum()
        return scores, logits

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the variational circuit and return attention weights.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (n_qubits, 3).
        entangle_params : np.ndarray
            Shape (n_qubits - 1,).
        inputs : np.ndarray
            Shape (n_qubits,).
        shots : int, default 1024
            Number of shots for the simulator.

        Returns
        -------
        np.ndarray
            Attention weights of shape (n_qubits,).
        """
        # Use the default qubit simulator with shot sampling
        dev = qml.device("default.qubit", wires=self.n_qubits, shots=shots)

        @qml.qnode(dev)
        def circuit():
            for i in range(self.n_qubits):
                qml.RX(inputs[i] * np.pi, wires=i)
            for i in range(self.n_qubits):
                qml.RX(rotation_params[i, 0], wires=i)
                qml.RY(rotation_params[i, 1], wires=i)
                qml.RZ(rotation_params[i, 2], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        raw_expectations = circuit()
        logits = np.array(raw_expectations)
        logits = (logits + 1) / 2
        weights = logits / logits.sum()
        return weights

__all__ = ["QuantumSelfAttention"]
