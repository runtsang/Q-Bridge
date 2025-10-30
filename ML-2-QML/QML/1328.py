"""Variational quantum self‑attention using Pennylane."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class SelfAttention:
    """
    Quantum self‑attention block implemented with a parameter‑shaped variational circuit.
    The circuit encodes the input sequence as angle‑encoded states, applies a
    strongly entangling layer per “head”, and measures expectation values to
    produce attention‑like outputs.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, shots: int = 1024):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=embed_dim, shots=shots)

    def _variational_layer(self, params: np.ndarray):
        """Strongly entangling layer with per‑wire rotations."""
        qml.templates.StronglyEntanglingLayers(params, wires=range(self.embed_dim))

    def _qnode(self, inputs: np.ndarray, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """QNode that returns a vector of expectation values."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            # Angle‑encoding of inputs
            for i, val in enumerate(inputs):
                qml.RY(val, wires=i)

            # Apply rotation parameters as a variational layer
            self._variational_layer(rotation_params)

            # Entanglement layer – one controlled‑RZ per adjacent pair
            for i in range(self.embed_dim - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Measure expectation of PauliZ on each wire
            return [qml.expval(qml.PauliZ(i)) for i in range(self.embed_dim)]

        return circuit()

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
            Parameters for the variational layer. Shape: (num_layers, embed_dim, 3)
        entangle_params : np.ndarray
            Parameters for the controlled‑RZ entanglement. Shape: (embed_dim-1,)
        inputs : np.ndarray
            Input sequence of shape (seq_len, embed_dim). Each row is encoded.

        Returns
        -------
        np.ndarray
            Quantum‑derived attention vector of shape (seq_len, embed_dim).
        """
        seq_len = inputs.shape[0]
        outputs = []
        for i in range(seq_len):
            out = self._qnode(
                inputs[i], rotation_params.reshape(-1), entangle_params
            )
            outputs.append(out)
        return np.array(outputs)
