from __future__ import annotations
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute

class HybridSelfAttention:
    """
    Quantum hybrid self‑attention that mirrors the classical interface.
    A circuit is built from rotation and entanglement parameters to
    encode the attention pattern, followed by a parameterized
    fully‑connected sub‑circuit that emulates a dense layer.
    """
    def __init__(self, n_qubits: int = 4, fc_qubits: int = 1):
        self.n_qubits = n_qubits
        self.fc_qubits = fc_qubits
        self.qr = QuantumRegister(n_qubits + fc_qubits, "q")
        self.cr = ClassicalRegister(n_qubits + fc_qubits, "c")
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray,
                       fc_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)

        # Attention block
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)

        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)

        # Fully‑connected sub‑circuit
        for i, theta in enumerate(fc_params):
            qc.ry(theta, self.n_qubits + i)

        qc.measure_all()
        return qc

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            fc_params: np.ndarray = None,
            shots: int = 1024,
            backend=None) -> np.ndarray:
        """
        Execute the hybrid circuit and return a single‑dimensional
        expectation value derived from measurement counts.
        """
        if fc_params is None:
            fc_params = np.zeros(self.fc_qubits)
        if backend is None:
            backend = self.backend
        qc = self._build_circuit(rotation_params, entangle_params, fc_params)
        job = execute(qc, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        probs = np.array(list(counts.values())) / shots
        states = np.array([int(s, 2) for s in counts.keys()], dtype=float)
        expectation = np.sum(states * probs)
        return np.array([expectation])

__all__ = ["HybridSelfAttention"]
