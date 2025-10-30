"""Quantum self‑attention built with Pennylane, providing differentiable attention weights."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class SelfAttention:
    """
    Variational quantum self‑attention that mirrors the classical API.
    Uses a parameter‑shaped rotation and entanglement circuit per head.
    """
    def __init__(self, n_qubits: int = 4, n_heads: int = 2, device: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.n_heads = n_heads
        self.device = qml.device(device, wires=n_qubits)
        # Create a separate QNode for each head
        self.qnodes = [self._make_qnode() for _ in range(n_heads)]

    def _make_qnode(self):
        @qml.qnode(self.device, interface="autograd")
        def circuit(params, inputs):
            """
            params: shape (n_qubits * 3 + n_qubits - 1,)
            inputs: shape (n_qubits,)
            """
            # Encode inputs as Z rotations
            for i, val in enumerate(inputs):
                qml.RZ(val, wires=i)

            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(params[3 * i], wires=i)
                qml.RY(params[3 * i + 1], wires=i)
                qml.RZ(params[3 * i + 2], wires=i)

            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CRX(params[self.n_qubits * 3 + i], wires=[i, i + 1])

            # Expectation of Z on each qubit gives a score
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (n_heads, n_qubits * 3), rotation angles for each head.
        entangle_params : np.ndarray
            Shape (n_heads, n_qubits - 1), entanglement angles for each head.
        inputs : np.ndarray
            Shape (batch, seq_len, n_qubits) – each token is a vector of qubit values.

        Returns
        -------
        np.ndarray
            Attention‑weighted representations, shape (batch, seq_len, n_qubits).
        """
        batch, seq_len, _ = inputs.shape
        outputs = []

        for b in range(batch):
            batch_out = []
            for t in range(seq_len):
                token = inputs[b, t]
                head_outputs = []

                for h in range(self.n_heads):
                    params = np.concatenate(
                        [rotation_params[h], entangle_params[h]]
                    )
                    # Run QNode
                    scores = self.qnodes[h](params, token)
                    # Convert to probabilities via softmax
                    probs = np.exp(scores) / np.sum(np.exp(scores))
                    head_outputs.append(probs)

                # Average over heads
                head_mean = np.mean(head_outputs, axis=0)
                batch_out.append(head_mean)

            outputs.append(batch_out)

        return np.array(outputs)
