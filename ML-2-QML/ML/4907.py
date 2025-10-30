"""Hybrid self‑attention that optionally leverages a quantum module for richer attention weights.

The module mirrors the classical SelfAttention helper but adds a flag to replace the softmax‑based scores
with a quantum‑generated probability distribution.  The quantum module can produce a distribution via a
parameterised Qiskit circuit (QCNN‑style) or, if provided, a SamplerQNN/EstimatorQNN object.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any


class _HybridSelfAttention:
    """Classical self‑attention with optional quantum augmentation."""

    def __init__(
        self,
        embed_dim: int,
        use_quantum: bool = False,
        quantum_module: Optional[Any] = None,
    ) -> None:
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the feature vectors.
        use_quantum : bool, default=False
            When True the `quantum_module` is queried for attention scores.
        quantum_module : Any, optional
            Object exposing a ``run`` method that returns a probability
            vector of shape ``(embed_dim,)``.  Typical choices are the
            ``HybridQuantumSelfAttention`` class defined in the QML module.
        """
        self.embed_dim = embed_dim
        self.use_quantum = use_quantum
        self.quantum_module = quantum_module

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Apply self‑attention to ``inputs``.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the query/key linear projections.
        entangle_params : np.ndarray
            Parameters for the key/value linear projections.
        inputs : np.ndarray
            Input matrix of shape ``(batch, embed_dim)``.

        Returns
        -------
        np.ndarray
            Weighted sum of the values, shape ``(batch, embed_dim)``.
        """
        # Classical query, key, value
        query = torch.tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.tensor(inputs, dtype=torch.float32)

        # Classical attention scores
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)

        # Quantum‑augmented scores
        if self.use_quantum and self.quantum_module is not None:
            q_counts = self.quantum_module.run(
                rotation_params, entangle_params
            )
            # Convert counts to probabilities
            total = sum(q_counts.values())
            if total > 0:
                q_probs = np.array(
                    [q_counts.get(str(i), 0) / total for i in range(self.embed_dim)],
                    dtype=np.float32,
                )
                # Blend classical and quantum scores (simple 50/50 mix)
                scores = 0.5 * scores + 0.5 * torch.tensor(q_probs, dtype=torch.float32)

        # Weighted sum
        return (scores @ value).numpy()


def HybridSelfAttention(
    embed_dim: int,
    use_quantum: bool = False,
    quantum_module: Optional[Any] = None,
) -> _HybridSelfAttention:
    """Factory returning a configured :class:`_HybridSelfAttention` instance."""
    return _HybridSelfAttention(embed_dim, use_quantum, quantum_module)


__all__ = ["HybridSelfAttention"]
