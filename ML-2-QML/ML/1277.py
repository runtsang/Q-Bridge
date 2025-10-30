"""Hybrid classical‑quantum self‑attention module.

The class SelfAttention__gen226 first computes classical attention scores
using query/key/value matrices.  It then obtains a quantum‑derived mask
from the quantum module SelfAttention__gen226 (implemented with Pennylane)
and applies it to the attention output.  The interface remains the same
as the original seed: ``run(inputs, rotation_params, entangle_params)``.
"""

import numpy as np
import torch
from SelfAttention__gen226_qml import SelfAttention__gen226 as QuantumSelfAttention

class SelfAttention__gen226:
    """Hybrid self‑attention with quantum‑based mask."""

    def __init__(self, embed_dim: int, n_qubits: int = None):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the input embeddings.
        n_qubits : int, optional
            Number of qubits for the quantum circuit.  If None, it is
            automatically set to ceil(log2(embed_dim)).
        """
        self.embed_dim = embed_dim
        if n_qubits is None:
            n_qubits = int(np.ceil(np.log2(embed_dim)))
        self.quantum_attention = QuantumSelfAttention(n_qubits)

    def run(self, inputs: np.ndarray, rotation_params: np.ndarray,
            entangle_params: np.ndarray) -> np.ndarray:
        """
        Compute the hybrid attention output.

        Parameters
        ----------
        inputs : np.ndarray
            Input embeddings of shape (batch, seq_len, embed_dim).
        rotation_params : np.ndarray
            Parameters for the single‑qubit rotations in the quantum circuit.
        entangle_params : np.ndarray
            Parameters for the controlled‑RZ gates in the quantum circuit.

        Returns
        -------
        np.ndarray
            The attended output of shape (batch, seq_len, embed_dim).
        """
        # Classical attention
        batch, seq_len, _ = inputs.shape
        queries = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32)
        keys = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32)
        scores = torch.softmax(queries @ keys.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1)
        # Quantum mask
        mask = self.quantum_attention.run(rotation_params, entangle_params)
        # Ensure mask shape matches seq_len
        mask = mask[:seq_len]
        mask = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        # Apply mask to scores
        masked_scores = scores * mask
        # Weighted sum over values
        values = torch.as_tensor(inputs, dtype=torch.float32)
        output = masked_scores @ values
        return output.numpy()

__all__ = ["SelfAttention__gen226"]
