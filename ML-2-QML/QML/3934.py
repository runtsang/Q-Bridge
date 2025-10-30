"""Hybrid self‑attention with parameterized quantum circuit and sampler.

The quantum block builds a rotation/entanglement circuit, measures it,
and uses a state‑vector sampler to produce attention probabilities.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as StatevectorSampler

class HybridSelfAttentionQML:
    """
    Quantum self‑attention block that mirrors the classical interface.
    """
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.sampler = StatevectorSampler(self.backend)

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr, self.cr)
        # Rotation layer
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            circ.cx(i, i + 1)
        circ.measure(self.qr, self.cr)
        return circ

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit and return a probability‑weighted attention matrix.
        Args:
            rotation_params: Array of shape (3*n_qubits,)
            entangle_params: Array of shape (n_qubits-1,) – unused but kept for API
            inputs: Array of shape (batch, seq_len, embed_dim) – mapped to rotation angles
            shots: Number of shots for measurement
        Returns:
            Attention matrix of shape (batch, seq_len, seq_len)
        """
        # Map input vectors to rotation parameters for each batch element
        batch, seq_len, embed_dim = inputs.shape
        attn_matrices = np.zeros((batch, seq_len, seq_len))
        for b in range(batch):
            for s in range(seq_len):
                # Use the s‑th embedding to perturb rotation_params
                perturbed = rotation_params + inputs[b, s] * 0.1
                circ = self._build_circuit(perturbed, entangle_params)
                job = qiskit.execute(circ, self.backend, shots=shots)
                counts = job.result().get_counts(circ)
                probs = np.array([counts.get(bin(i)[2:].zfill(self.n_qubits), 0) for i in range(2**self.n_qubits)])
                probs = probs / probs.sum()
                # Interpret probability distribution as attention logits
                logits = probs.reshape((self.n_qubits, self.n_qubits))
                attn_matrices[b, s] = logits
        return attn_matrices

__all__ = ["HybridSelfAttentionQML"]
