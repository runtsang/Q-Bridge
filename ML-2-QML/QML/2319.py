import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit

class HybridSelfAttention:
    """
    Quantum hybrid self‑attention module.
    Encodes each input patch via a 2×2 quanv circuit, then applies a
    variational attention circuit on the concatenated qubits.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 127, shots: int = 1024):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def _quantum_patch(self, data_patch: np.ndarray) -> QuantumCircuit:
        """
        Build a quanv circuit for a single 2×2 patch.
        data_patch: (kernel_size, kernel_size) array of pixel values.
        Returns a quantum circuit that measures probability of |1>.
        """
        n_qubits = self.kernel_size ** 2
        qc = QuantumCircuit(n_qubits, n_qubits)
        theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(n_qubits, 2)
        qc.measure_all()
        param_bind = {theta[i]: np.pi if data_patch.flat[i] > self.threshold else 0.0
                      for i in range(n_qubits)}
        qc = qc.bind_parameters(param_bind)
        return qc

    def run(self, data: np.ndarray, rotation_params: np.ndarray,
            entangle_params: np.ndarray, shots: int = None) -> np.ndarray:
        """
        Execute hybrid attention on a batch of sequences.
        data: (batch, seq_len, kernel_size, kernel_size)
        rotation_params: (seq_len, n_qubits, 3) rotation angles for attention circuit
        entangle_params: (seq_len-1, n_qubits-1) parameters for CNOT‑like entanglement
        Returns: (batch, seq_len, seq_len) attention probabilities.
        """
        if shots is None:
            shots = self.shots
        batch, seq_len, _, _ = data.shape
        n_qubits = self.kernel_size ** 2

        # Pre‑compute quanv circuits for each patch and run them
        features = np.zeros((batch, seq_len))
        for b in range(batch):
            for s in range(seq_len):
                qc = self._quantum_patch(data[b, s])
                job = qiskit.execute(qc, self.backend, shots=shots)
                result = job.result().get_counts(qc)
                prob = 0.0
                for key, val in result.items():
                    ones = sum(int(bit) for bit in key)
                    prob += ones * val
                features[b, s] = prob / (shots * n_qubits)

        # Build attention circuit using the extracted features as rotation angles
        attn_circuits = []
        for b in range(batch):
            qc = QuantumCircuit(seq_len, seq_len)
            for i in range(seq_len):
                angle = rotation_params[i, 0, 0] * features[b, i]
                qc.rx(angle, i)
                qc.ry(angle, i)
                qc.rz(angle, i)
            for i in range(seq_len - 1):
                qc.cx(i, i + 1)
            qc.measure_all()
            attn_circuits.append(qc)

        # Execute attention circuits and compute probabilities
        attn_probs = np.zeros((batch, seq_len, seq_len))
        for b in range(batch):
            job = qiskit.execute(attn_circuits[b], self.backend, shots=shots)
            result = job.result().get_counts(attn_circuits[b])
            for key, val in result.items():
                probs = np.array([int(bit) for bit in key]) * val
                attn_probs[b] += probs.reshape(seq_len, 1)
        attn_probs /= (shots * seq_len)
        return attn_probs
