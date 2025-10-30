"""Quantum hybrid self‑attention using a variational circuit and a quantum kernel.

The class extends the original SelfAttention interface but replaces the
classical RBF kernel with a quantum kernel evaluated via statevector
overlaps. It encodes each input as a set of rotation angles, applies a
variational entanglement layer, and computes the squared inner product
between encoded query and key states. The resulting similarity matrix
is softmaxed to produce attention weights, which are applied to the
value vectors. The implementation uses Qiskit for circuit construction
and statevector extraction, and is fully compatible with any Qiskit
backend.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import torch

class HybridSelfAttention:
    """Quantum self‑attention with kernel‑based similarity."""
    def __init__(self, n_qubits: int = 4, gamma: float = 1.0):
        self.n_qubits = n_qubits
        self.gamma = gamma
        self.entangle_params = np.zeros(n_qubits - 1)

    def _encode_statevector(self, params: np.ndarray) -> Statevector:
        circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            circuit.rx(params[3 * i], i)
            circuit.ry(params[3 * i + 1], i)
            circuit.rz(params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(self.entangle_params[i], i, i + 1)
        return Statevector.from_instruction(circuit)

    def _kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        sv_x = self._encode_statevector(x)
        sv_y = self._encode_statevector(y)
        return abs(np.vdot(sv_x.data, sv_y.data)) ** 2

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        # Store entangle_params for encoding
        self.entangle_params = entangle_params
        # Assume rotation_params shape: (batch, 3 * n_qubits)
        batch = rotation_params.shape[0]
        # Compute kernel matrix
        kernel_matrix = np.zeros((batch, batch))
        for i in range(batch):
            for j in range(batch):
                kernel_matrix[i, j] = self._kernel(rotation_params[i], rotation_params[j])
        # Softmax to get attention weights
        attn_weights = np.exp(kernel_matrix - np.max(kernel_matrix, axis=-1, keepdims=True))
        attn_weights /= attn_weights.sum(axis=-1, keepdims=True)
        # Value vectors are just the inputs (rotation_params)
        values = rotation_params
        # Weighted sum
        out = attn_weights @ values
        return out

__all__ = ["HybridSelfAttention"]
