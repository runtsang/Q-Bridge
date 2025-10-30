"""Hybrid quantum self‑attention module that emulates the classical interface.
The implementation builds a Qiskit circuit that applies rotations derived from the
rotation_params, entangles qubits with parameters from entangle_params, and measures
to obtain an attention distribution.  The returned counts are converted to a probability
matrix that is used as attention weights over the input values.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class HybridSelfAttentionKernel:
    """Quantum self‑attention circuit that mirrors the classical API."""
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        counts = job.result().get_counts(circuit)
        # Convert counts to probability distribution
        probs = np.zeros(2 ** self.n_qubits)
        for bitstring, cnt in counts.items():
            idx = int(bitstring[::-1], 2)
            probs[idx] = cnt / shots
        # Reshape to an attention matrix (n_qubits × n_qubits) by grouping
        # each qubit's probability as a row (simple heuristic)
        return probs.reshape(self.n_qubits, -1)

def HybridSelfAttentionKernel_factory():
    """Factory that returns a quantum self‑attention object with a pre‑selected backend."""
    backend = Aer.get_backend("qasm_simulator")
    return HybridSelfAttentionKernel(n_qubits=4)
