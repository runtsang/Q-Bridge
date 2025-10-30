import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class SelfAttention:
    """Parameter‑efficient variational quantum self‑attention."""

    def __init__(self, n_qubits: int, heads: int = 2):
        self.n_qubits = n_qubits
        self.heads = heads
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """
        Build a lightweight variational circuit.

        rotation_params : shape (n_qubits * 3,)
        entangle_params : shape (n_qubits - 1,)
        """
        circuit = QuantumCircuit(self.qr, self.cr)
        # Single‑qubit rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entangling layer (CNOT chain)
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
        # Parameterised CRX entanglement
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit and return attention weights.

        Returns
        -------
        np.ndarray
            Attention matrix of shape (heads, n_qubits, n_qubits).
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)

        # Convert counts to probability distribution over bitstrings
        probs = {}
        for bitstring, cnt in counts.items():
            probs[bitstring] = cnt / shots

        # Compute marginal probabilities for each qubit
        marginals = np.zeros(self.n_qubits)
        for bitstring, p in probs.items():
            bits = [int(b) for b in bitstring[::-1]]  # reverse due to qiskit ordering
            marginals += np.array(bits) * p

        # Build attention matrices per head
        attention_matrices = []
        for h in range(self.heads):
            # For each head, use a simple outer product of marginals
            attn = np.outer(marginals, marginals)
            attention_matrices.append(attn)

        # Stack heads
        return np.stack(attention_matrices, axis=0)  # shape (heads, n_qubits, n_qubits)
