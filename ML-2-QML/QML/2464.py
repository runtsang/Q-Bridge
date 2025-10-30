"""Hybrid self‑attention implemented in Qiskit.

The quantum implementation mirrors the classical RBF‑kernel attention
by using a swap‑test quantum kernel to compute similarity between
query and key registers.  The interface is identical to the classical
module, enabling easy back‑end switching.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import AerSimulator

class HybridSelfAttention:
    """
    Quantum self‑attention that uses a swap‑test kernel to compute
    attention scores between query and key states.
    """
    def __init__(self, n_qubits: int = 4, gamma: float = 1.0):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits per register.  For a single attention head
            we use two registers of ``n_qubits`` qubits each.
        gamma : float
            Hyper‑parameter controlling the width of the kernel
            (used in the variational circuit).
        """
        self.n_qubits = n_qubits
        self.gamma = gamma
        self.backend = AerSimulator()

    def _encode_vector(self, qc: QuantumCircuit, vec: np.ndarray, offset: int = 0) -> QuantumCircuit:
        """
        Apply Ry rotations to encode the vector ``vec`` into ``qc``.
        The first ``n_qubits`` qubits are used for the vector.
        """
        for i, val in enumerate(vec):
            qc.ry(val, i + offset)
        return qc

    def _kernel_circuit(self, x: np.ndarray, y: np.ndarray) -> QuantumCircuit:
        """
        Build a circuit that prepares |x> and |y> on separate registers
        and measures the overlap |<x|y>|^2 via a swap test.
        """
        qr_x = QuantumRegister(self.n_qubits, "x")
        qr_y = QuantumRegister(self.n_qubits, "y")
        qr_s = QuantumRegister(1, "s")   # ancilla for swap test
        cr = ClassicalRegister(1, "c")

        qc = QuantumCircuit(qr_s, qr_x, qr_y, cr)

        # Encode x on qr_x
        qc = self._encode_vector(qc, x, offset=0)
        # Encode y on qr_y
        qc = self._encode_vector(qc, y, offset=self.n_qubits)

        # Swap test
        qc.h(qr_s[0])
        for i in range(self.n_qubits):
            qc.cswap(qr_s[0], qr_x[i], qr_y[i])
        qc.h(qr_s[0])
        qc.measure(qr_s[0], cr[0])

        return qc

    def _measure_kernel(self, x: np.ndarray, y: np.ndarray, shots: int = 1024) -> float:
        """
        Execute the swap‑test kernel circuit and return the estimated
        overlap probability.
        """
        qc = self._kernel_circuit(x, y)
        job = self.backend.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        p0 = counts.get("0", 0) / shots
        return p0

    def run(self,
            inputs: np.ndarray,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the quantum self‑attention on a batch of inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Input embeddings of shape (batch, seq_len, embed_dim).
        rotation_params : np.ndarray
            Parameters used to rotate the input before projection.
        entangle_params : np.ndarray
            Parameters used to entangle the input before projection.
        shots : int, optional
            Number of shots for the kernel evaluation.

        Returns
        -------
        np.ndarray
            Output embeddings of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, embed_dim = inputs.shape
        rot = rotation_params.reshape(self.n_qubits, -1)
        ent = entangle_params.reshape(self.n_qubits, -1)

        outputs = np.zeros_like(inputs)

        for b in range(batch):
            for i in range(seq_len):
                q_vec = np.dot(inputs[b, i], rot)
                k_vecs = np.dot(inputs[b], ent)  # (seq_len, n_qubits)
                scores = np.array([self._measure_kernel(q_vec, k_vec, shots)
                                   for k_vec in k_vecs])
                scores = scores / scores.sum()
                outputs[b, i] = np.dot(scores, inputs[b])
        return outputs

    def kernel_matrix(self,
                      a: np.ndarray,
                      b: np.ndarray,
                      shots: int = 1024) -> np.ndarray:
        """
        Compute the Gram matrix between two batches ``a`` and ``b`` using
        the swap‑test quantum kernel.
        """
        n_a, n_b = a.shape[0], b.shape[0]
        gram = np.zeros((n_a, n_b))
        for i in range(n_a):
            for j in range(n_b):
                gram[i, j] = self._measure_kernel(a[i], b[j], shots)
        return gram

__all__ = ["HybridSelfAttention"]
