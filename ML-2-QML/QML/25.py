"""Variational quantum self‑attention using Pennylane."""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class SelfAttentionEnhanced:
    """
    Quantum‑classical hybrid self‑attention.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (each qubit represents one embedding dimension).
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(rotation_params, entangle_params):
            # Parameterized rotations per qubit
            for i in range(n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Entangling layer (CZ gates)
            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i + 1])

            # Measure expectation values of Z to form attention logits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def _softmax(self, logits):
        logits = pnp.asarray(logits)
        exp = pnp.exp(logits - pnp.max(logits))
        return exp / pnp.sum(exp)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Execute the quantum self‑attention circuit and return a processed output.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles for the RX/RY/RZ gates (length 3 * n_qubits).
        entangle_params : np.ndarray
            Entangling gate parameters (unused in this simple circuit but kept for API compatibility).
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, n_qubits). Each example is encoded as a binary string
            by thresholding the continuous values at 0.5.

        Returns
        -------
        np.ndarray
            Output tensor of shape (batch, seq_len, n_qubits) where the quantum circuit has produced
            a probabilistic attention weight vector per token.
        """
        batch, seq_len, _ = inputs.shape
        output = np.zeros_like(inputs)

        for b in range(batch):
            for t in range(seq_len):
                # Encode input token as binary string
                binary = (inputs[b, t] > 0.5).astype(float)
                # Feed binary string into the circuit as the initial state
                self.dev.reset()
                for i, val in enumerate(binary):
                    if val == 1.0:
                        qml.PauliX(wires=i)

                logits = self.circuit(rotation_params, entangle_params)
                attn_weights = self._softmax(logits)
                output[b, t] = attn_weights

        return output


__all__ = ["SelfAttentionEnhanced"]
