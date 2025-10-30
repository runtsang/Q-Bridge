"""Quantum‑enhanced self‑attention that uses a parameterised circuit, a quantum kernel via a swap‑test and a Qiskit sampler.

The implementation keeps the same public API as the classical version so it can be dropped in as a drop‑in replacement.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.primitives import Sampler as QiskitSampler

class HybridSelfAttentionQuantum:
    """Quantum self‑attention with optional quantum kernel and sampler."""
    def __init__(self, n_qubits: int = 4, backend=None):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.sampler = QiskitSampler(backend=self.backend)

    def _encode_vector(self, circuit: QuantumCircuit, vec: np.ndarray, offset: int = 0) -> None:
        """Encode a classical vector into rotation angles on the first n_qubits qubits."""
        for i, val in enumerate(vec):
            circuit.ry(val, offset + i)

    def _build_attention_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> QuantumCircuit:
        """Build a circuit that applies the attention logic to the input data."""
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Encode each input vector as rotations
        for i, vec in enumerate(inputs):
            self._encode_vector(circuit, vec, offset=i)

        # Apply rotation parameters (gate angles)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entangle neighbouring qubits
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        circuit.measure(qr, cr)
        return circuit

    def _swap_test(self, x: np.ndarray, y: np.ndarray) -> float:
        """Estimate the overlap between two states |x⟩ and |y⟩ via a swap‑test."""
        anc = QuantumRegister(1, "a")
        qr_x = QuantumRegister(self.n_qubits, "x")
        qr_y = QuantumRegister(self.n_qubits, "y")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(anc, qr_x, qr_y, cr)

        # Prepare |x⟩ and |y⟩
        for i, val in enumerate(x):
            circuit.ry(val, qr_x[i])
        for i, val in enumerate(y):
            circuit.ry(val, qr_y[i])

        circuit.h(anc[0])
        for i in range(self.n_qubits):
            circuit.cswap(anc[0], qr_x[i], qr_y[i])
        circuit.h(anc[0])
        circuit.measure(anc, cr)

        result = execute(circuit, self.backend, shots=1024).result()
        counts = result.get_counts(circuit)
        prob_0 = counts.get("0", 0) / 1024
        # Fidelity estimate: F = (1 + P(0)) / 2
        return (1 + prob_0) / 2

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Args:
            inputs: 2‑D array of shape (seq_len, embed_dim)
            rotation_params: array of shape (3 * n_qubits,)
            entangle_params: array of shape (n_qubits - 1,)
        Returns:
            numpy array of shape (seq_len, embed_dim) – the attention‑weighted sum.
        """
        seq_len, embed_dim = inputs.shape
        # Build the attention circuit and execute
        circuit = self._build_attention_circuit(rotation_params, entangle_params, inputs)
        job = execute(circuit, self.backend, shots=shots)
        counts = job.result().get_counts(circuit)

        # Convert counts to a probability distribution over basis states
        probs = np.zeros(2 ** self.n_qubits)
        for bitstring, cnt in counts.items():
            idx = int(bitstring[::-1], 2)  # Qiskit stores bits reversed
            probs[idx] = cnt / shots

        # Use the probability distribution as attention weights
        # Reshape to (seq_len, embed_dim) by assuming each basis state corresponds to a key
        # For demonstration we simply take the first seq_len probabilities
        attn_weights = probs[:seq_len]
        attn_weights = attn_weights / attn_weights.sum()

        # Compute weighted sum of the input vectors
        attn_output = np.tensordot(attn_weights, inputs, axes=1)

        # Optionally refine the weights using a quantum kernel (swap‑test)
        kernel_matrix = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                kernel_matrix[i, j] = self._swap_test(inputs[i], inputs[j])
        kernel_norm = kernel_matrix / kernel_matrix.sum(axis=1, keepdims=True)
        refined_weights = kernel_norm @ attn_weights
        refined_weights = refined_weights / refined_weights.sum()

        # Final weighted sum
        final_output = np.tensordot(refined_weights, inputs, axes=1)
        return final_output

__all__ = ["HybridSelfAttentionQuantum"]
