"""Quantum self‑attention module that computes attention scores via a quantum kernel.

The module encodes each embedded patch into a 2‑qubit state, evaluates pairwise
overlaps using a state‑vector simulator, and derives soft‑maxed attention
weights.  The interface mirrors the classical SelfAttention class so that
either back‑end can be swapped at runtime.
"""

import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.quantum_info import Statevector

class SelfAttention:
    """
    Quantum self‑attention.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings (must be a power of two).
    backend : str, optional
        Name of the Aer backend to use. Defaults to'statevector_simulator'.
    """

    def __init__(self, embed_dim: int, backend: str = "statevector_simulator") -> None:
        self.embed_dim = embed_dim
        if not (embed_dim & (embed_dim - 1) == 0):
            raise ValueError("embed_dim must be a power of two for amplitude encoding.")
        self.n_qubits = int(np.log2(embed_dim))
        self.backend = backend
        self.qi = QuantumInstance(Aer.get_backend(self.backend))

    def _pad_vector(self, vec: np.ndarray) -> np.ndarray:
        """Pad a vector to length 2^n_qubits."""
        size = 1 << self.n_qubits
        padded = np.zeros(size, dtype=complex)
        padded[: len(vec)] = vec
        return padded

    def _statevector(self, vec: np.ndarray) -> Statevector:
        """Return the normalized statevector for a given vector."""
        padded = self._pad_vector(vec)
        norm = np.linalg.norm(padded)
        if norm == 0:
            raise ValueError("Zero‑norm vector cannot be encoded.")
        return Statevector(padded / norm)

    def _kernel_matrix(self, queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """
        Compute the quantum kernel matrix K_{ij} = |<q_i|k_j>|^2.
        """
        n_q = queries.shape[0]
        n_k = keys.shape[0]
        K = np.empty((n_q, n_k), dtype=float)
        for i in range(n_q):
            q_vec = self._statevector(queries[i])
            for j in range(n_k):
                k_vec = self._statevector(keys[j])
                overlap = np.vdot(q_vec, k_vec)
                K[i, j] = abs(overlap) ** 2
        return K

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the quantum self‑attention.

        Parameters
        ----------
        rotation_params, entangle_params : np.ndarray
            Placeholder arguments kept for API compatibility. They are
            currently unused but allow future extensions where a variational
            circuit might be employed.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).
        shots : int, optional
            Number of shots for the state‑vector simulator (ignored).
        Returns
        -------
        np.ndarray
            Attention‑weighted sum of the inputs, shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, embed_dim = inputs.shape
        if embed_dim!= self.embed_dim:
            raise ValueError("Input embed_dim does not match the instantiated value.")

        # Compute quantum kernel for each example in the batch
        outputs = np.empty_like(inputs)
        for b in range(batch):
            queries = inputs[b]  # (seq_len, embed_dim)
            keys = queries  # self‑attention
            K = self._kernel_matrix(queries, keys)  # (seq_len, seq_len)
            # Softmax over keys
            weights = np.exp(K / np.sqrt(self.embed_dim))
            weights /= weights.sum(axis=1, keepdims=True)
            # Weighted sum of values
            values = inputs[b]  # (seq_len, embed_dim)
            outputs[b] = weights @ values
        return outputs

__all__ = ["SelfAttention"]
