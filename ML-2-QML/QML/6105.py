"""Quantum self‑attention with multi‑head support."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute


class QuantumSelfAttention:
    """
    Quantum multi‑head self‑attention.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, optional
        Number of attention heads. Must divide embed_dim.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1):
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.n_qubits_per_head = self.head_dim
        self.total_qubits = self.n_qubits_per_head * num_heads

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        head_idx: int,
    ) -> QuantumCircuit:
        """
        Build a circuit for a single attention head.

        Parameters
        ----------
        rotation_params : ndarray of shape (head_dim, head_dim)
            Rotation angles for each qubit.
        entangle_params : ndarray of shape (head_dim - 1,)
            Entangling angles between consecutive qubits.
        head_idx : int
            Index of the head (0‑based).

        Returns
        -------
        QuantumCircuit
        """
        qr = QuantumRegister(self.n_qubits_per_head, f"q_{head_idx}")
        cr = ClassicalRegister(self.n_qubits_per_head, f"c_{head_idx}")
        qc = QuantumCircuit(qr, cr)

        # Single‑qubit rotations
        for i in range(self.n_qubits_per_head):
            qc.rx(rotation_params[i, 0], i)
            qc.ry(rotation_params[i, 1], i)
            qc.rz(rotation_params[i, 2], i)

        # Simple nearest‑neighbour entanglement
        for i in range(self.n_qubits_per_head - 1):
            qc.crx(entangle_params[i], i, i + 1)

        qc.measure(qr, cr)
        return qc

    def run(
        self,
        backend,
        rotation_params_list: list[np.ndarray],
        entangle_params_list: list[np.ndarray],
        shots: int = 1024,
    ) -> list[dict]:
        """
        Execute quantum attention for all heads.

        Parameters
        ----------
        backend : qiskit.providers.Backend
        rotation_params_list : list of ndarray
            One array per head of shape (head_dim, head_dim).
        entangle_params_list : list of ndarray
            One array per head of shape (head_dim - 1,).
        shots : int, optional

        Returns
        -------
        list of dict
            Measurement counts per head.
        """
        if len(rotation_params_list)!= self.num_heads or len(entangle_params_list)!= self.num_heads:
            raise ValueError("Parameter lists must match num_heads.")

        counts_per_head = []
        for idx in range(self.num_heads):
            qc = self._build_circuit(rotation_params_list[idx], entangle_params_list[idx], idx)
            job = execute(qc, backend, shots=shots)
            result = job.result()
            counts_per_head.append(result.get_counts(qc))
        return counts_per_head


__all__ = ["QuantumSelfAttention"]
