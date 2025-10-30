"""Hybrid self‑attention combining a classical transformer head with a quantum‑guided attention block.

The class exposes the same interface as the original SelfAttention helper but
delegates the attention‑weight computation to a quantum sampler.  This
provides a tunable quantum contribution while keeping the rest of the
pipeline fully classical.
"""

import numpy as np
import torch
import torch.nn as nn

# Import the quantum implementation
from.SelfAttention__gen616_qml import HybridSelfAttention as QuantumHybridSelfAttention

class HybridSelfAttention:
    """Classical wrapper that uses a quantum sampler to produce attention weights."""
    def __init__(self, embed_dim: int, quantum_sampler: QuantumHybridSelfAttention | None = None):
        self.embed_dim = embed_dim
        self.quantum_head = quantum_sampler or QuantumHybridSelfAttention(n_qubits=embed_dim)
        # Linear projections for query, key, and value
        self.Wq = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Wk = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Wv = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute self‑attention with quantum‑generated weights.

        Parameters
        ----------
        rotation_params, entangle_params
            Parameters forwarded to the quantum sampler; they are interpreted
            as rotation angles for the underlying quantum circuit.
        inputs
            2‑D array of shape (seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            The attended representation of shape (seq_len, embed_dim).
        """
        # Classical linear projections
        q = torch.as_tensor(inputs @ self.Wq, dtype=torch.float32)
        k = torch.as_tensor(inputs @ self.Wk, dtype=torch.float32)
        v = torch.as_tensor(inputs @ self.Wv, dtype=torch.float32)

        seq_len = inputs.shape[0]
        # Prepare query‑key pairs for the quantum sampler
        qs = q.numpy().repeat(seq_len, axis=0)
        ks = np.tile(k.numpy(), (seq_len, 1))
        query_key_pairs = np.stack([qs, ks], axis=1).reshape(-1, 2)

        # Quantum attention logits
        raw_scores = self.quantum_head.run(
            rotation_params=rotation_params,
            entangle_params=entangle_params,
            inputs=query_key_pairs,
            shots=2048,
        )
        # raw_scores shape: (seq_len*seq_len, 2). Use probability of outcome '1'
        raw_scores = raw_scores[:, 1].reshape(seq_len, seq_len)
        # Normalise to obtain attention weights
        probs = raw_scores / raw_scores.sum(axis=1, keepdims=True)
        # Weighted sum over values
        attended = probs @ v.numpy()
        return attended

__all__ = ["HybridSelfAttention"]
