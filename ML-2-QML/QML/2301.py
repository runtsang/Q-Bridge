import qiskit
from qiskit import QuantumCircuit, Aer, transpile, assemble
import numpy as np

class HybridQuantumHybridNetQuantumCircuit:
    """
    Quantum circuits used by HybridQuantumHybridNet.
    Provides a two‑qubit variational head and an eight‑qubit attention block.
    """
    def __init__(self, n_qubits_head=2, n_qubits_attn=8, shots=200):
        self.backend = qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.head_circuit = self._build_head(n_qubits_head)
        self.attn_circuit = self._build_attention(n_qubits_attn)

    def _build_head(self, n_qubits):
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.ry(0.0, i)
        qc.measure_all()
        return qc

    def _build_attention(self, n_qubits):
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.ry(0.0, i)
        qc.measure_all()
        return qc

    def run_head(self, angles):
        """
        angles: 1‑D array of length n_qubits_head
        Returns expectation value of Pauli‑Z on qubit 0.
        """
        qc = self.head_circuit.copy()
        for i, a in enumerate(angles):
            qc.ry(a, i)
        compiled = transpile(qc, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        exp = 0
        for bitstring, cnt in counts.items():
            z = 1 if bitstring[-1] == '0' else -1
            exp += z * cnt
        return exp / self.shots

    def run_attention(self, angles):
        """
        angles: 1‑D array of length n_qubits_attn
        Returns expectation value of Pauli‑Z on qubit 0.
        """
        qc = self.attn_circuit.copy()
        for i, a in enumerate(angles):
            qc.ry(a, i)
        compiled = transpile(qc, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        exp = 0
        for bitstring, cnt in counts.items():
            z = 1 if bitstring[-1] == '0' else -1
            exp += z * cnt
        return exp / self.shots

__all__ = ["HybridQuantumHybridNetQuantumCircuit"]
