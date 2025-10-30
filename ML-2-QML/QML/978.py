"""Quantum self‑attention using a variational circuit.

The implementation builds a parameterised circuit that
acts like a self‑attention block and outputs a probability
distribution over the input indices.  The circuit can be
run on a simulator or a real device via Pennylane.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from typing import Tuple


class QuantumSelfAttention:
    """
    Variational self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be >= 1).
    num_layers : int
        Number of variational layers.
    backend : str
        Pennylane device name (e.g. ``default.qubit`` or a Qiskit backend).
    shots : int
        Number of shots for measurement.
    """

    def __init__(
        self,
        n_qubits: int,
        num_layers: int = 2,
        backend: str = "default.qubit",
        shots: int = 1024,
    ):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.backend = backend
        self.shots = shots
        self.dev = qml.device(backend, wires=n_qubits, shots=shots)

        # Parameter shapes
        self.rotation_shape = (num_layers, n_qubits, 3)  # RX,RZ,RXX
        self.entangle_shape = (num_layers, n_qubits - 1)  # CX between neighbours

    def _circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Construct the variational circuit and return the measurement
        probabilities over the computational basis.
        """
        @qml.qnode(self.dev)
        def circuit():
            # Input encoding – basis encoding of classical vector
            for i, val in enumerate(inputs):
                if val > 0.5:
                    qml.PauliX(i)

            # Variational layers
            for l in range(self.num_layers):
                for q in range(self.n_qubits):
                    qml.RY(rotation_params[l, q, 0], wires=q)
                    qml.RZ(rotation_params[l, q, 1], wires=q)
                    qml.RX(rotation_params[l, q, 2], wires=q)

                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])

            # Measurement
            return qml.probs(wires=range(self.n_qubits))

        return circuit()

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute the circuit and return a probability distribution
        over the input indices (attention weights) and the raw
        measurement counts.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (num_layers, n_qubits, 3).
        entangle_params : np.ndarray
            Shape (num_layers, n_qubits-1).  Currently unused but kept
            for API compatibility.
        inputs : np.ndarray
            Classical input vector of length n_qubits.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (attention_weights, raw_counts)
        """
        probs = self._circuit(rotation_params, entangle_params, inputs)
        # Map basis states to integer indices
        indices = np.arange(2 ** self.n_qubits)
        # Attention weights are the probability of each index
        attn_weights = probs
        return attn_weights, probs

    def attention_output(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a weighted sum of the value vectors using the
        attention probabilities produced by the circuit.

        Parameters
        ----------
        values : np.ndarray
            Array of shape (seq_len, embed_dim) where seq_len == 2**n_qubits.

        Returns
        -------
        np.ndarray
            Weighted sum of values.
        """
        attn_weights, _ = self.run(rotation_params, entangle_params, inputs)
        return (attn_weights[:, None] * values).sum(axis=0)


def SelfAttention(
    n_qubits: int = 4,
    num_layers: int = 2,
    backend: str = "default.qubit",
    shots: int = 1024,
) -> QuantumSelfAttention:
    """
    Factory that returns a ready‑to‑use quantum self‑attention block.

    The signature mirrors the original ``SelfAttention`` function but
    exposes additional variational parameters.
    """
    return QuantumSelfAttention(n_qubits, num_layers, backend, shots)


__all__ = ["SelfAttention", "QuantumSelfAttention"]
