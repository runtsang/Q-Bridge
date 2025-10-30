from __future__ import annotations
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import ZFeatureMap

class HybridQuantumSelfAttentionQCNN:
    """
    Quantum implementation of a hybrid self‑attention / QCNN ansatz.
    The circuit consists of a Z‑feature map, a hierarchy of two‑qubit
    convolution and pooling blocks, and a dense CRX attention layer
    that entangles every pair of qubits.
    """
    def __init__(self, n_qubits: int = 8, backend=None):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _conv_block(self, angles: np.ndarray, qubits: tuple[int, int]) -> QuantumCircuit:
        """Two‑qubit convolution unit."""
        qc = QuantumCircuit(self.qr[qubits[0]], self.qr[qubits[1]], name="conv")
        qc.rz(-np.pi / 2, qubits[1])
        qc.cx(qubits[1], qubits[0])
        qc.rz(angles[0], qubits[0])
        qc.ry(angles[1], qubits[1])
        qc.cx(qubits[0], qubits[1])
        qc.ry(angles[2], qubits[1])
        qc.cx(qubits[1], qubits[0])
        qc.rz(np.pi / 2, qubits[0])
        return qc

    def _pool_block(self, angles: np.ndarray, qubits: tuple[int, int]) -> QuantumCircuit:
        """Two‑qubit pooling unit."""
        qc = QuantumCircuit(self.qr[qubits[0]], self.qr[qubits[1]], name="pool")
        qc.rz(-np.pi / 2, qubits[1])
        qc.cx(qubits[1], qubits[0])
        qc.rz(angles[0], qubits[0])
        qc.ry(angles[1], qubits[1])
        qc.cx(qubits[0], qubits[1])
        qc.ry(angles[2], qubits[1])
        return qc

    def _attention_entangle(self, qc: QuantumCircuit, angles: np.ndarray) -> QuantumCircuit:
        """CRX entanglement between every pair of qubits to mimic self‑attention."""
        idx = 0
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                qc.crx(angles[idx], i, j)
                idx += 1
        return qc

    def _build_ansatz(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """Assemble the full QCNN‑style ansatz with an attention layer."""
        qc = QuantumCircuit(self.n_qubits, name="HybridQCNN")
        # Feature map
        feature_map = ZFeatureMap(self.n_qubits)
        qc.compose(feature_map, inplace=True)

        # Convolution–pooling hierarchy (three layers)
        pair_indices = [(i, i + 1) for i in range(0, self.n_qubits, 2)]
        idx = 0
        for _ in range(3):  # three layers
            # Convolution block
            for qubits in pair_indices:
                angles = rotation_params[idx:idx + 3]
                qc.append(self._conv_block(angles, qubits), qubits)
                idx += 3
            # Pooling block
            for qubits in pair_indices:
                angles = rotation_params[idx:idx + 3]
                qc.append(self._pool_block(angles, qubits), qubits)
                idx += 3

        # Attention entanglement
        qc = self._attention_entangle(qc, entangle_params)

        return qc

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        """Execute the circuit and return measurement counts."""
        circuit = self._build_ansatz(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

def SelfAttention() -> HybridQuantumSelfAttentionQCNN:
    """Factory returning the hybrid quantum self‑attention / QCNN ansatz."""
    return HybridQuantumSelfAttentionQCNN()
