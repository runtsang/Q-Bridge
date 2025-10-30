"""
Hybrid self‑attention module combining classical attention with a quantum‑generated phase mask.
"""

import numpy as np
import torch
from typing import Callable

class SelfAttentionHybrid:
    """
    Parameters
    ----------
    embed_dim : int
        Embedding dimension of the input representation.
    quantum_func : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        Function that accepts rotation_params, entangle_params and inputs, and returns a
        quantum‑derived phase vector of shape (embed_dim,).
    """

    def __init__(
        self,
        embed_dim: int,
        quantum_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ):
        self.embed_dim = embed_dim
        self.quantum_func = quantum_func

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute hybrid attention output.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the single‑qubit rotations in the quantum circuit.
            Shape (3*embed_dim,).
        entangle_params : np.ndarray
            Parameters for the two‑qubit entangling gates.
            Shape (embed_dim-1,).
        inputs : np.ndarray
            Input matrix of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            Hybrid attention output of shape (batch, embed_dim).
        """
        # Classical self‑attention
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)

        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        classical_out = scores @ value

        # Quantum‑derived phase mask
        quantum_phase = torch.as_tensor(
            self.quantum_func(rotation_params, entangle_params, inputs), dtype=torch.float32
        )

        # Apply phase mask to classical output (element‑wise multiplication)
        out = classical_out * quantum_phase

        return out.numpy()

__all__ = ["SelfAttentionHybrid"]
