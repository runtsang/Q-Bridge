import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

def _clip(value: float, bound: float) -> float:
    """Parameter clipping used in both quantum and classical branches."""
    return max(-bound, min(bound, value))

class QuantumSelfAttention:
    """Quantum analogue of scaled dot‑product attention."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.backend  = Aer.get_backend("qasm_simulator")

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, 'q')
        cr = ClassicalRegister(self.n_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        for i in range(self.n_qubits):
            qc.rx(_clip(rotation_params[3*i],   2*np.pi), qr[i])
            qc.ry(_clip(rotation_params[3*i+1], 2*np.pi), qr[i])
            qc.rz(_clip(rotation_params[3*i+2], 2*np.pi), qr[i])
        for i in range(self.n_qubits-1):
            qc.crx(_clip(entangle_params[i], 2*np.pi), qr[i], qr[i+1])
        qc.measure(qr, cr)
        return qc

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        qc = self._build_circuit(rotation_params, entangle_params)
        job = execute(qc, self.backend, shots=shots)
        counts = job.result().get_counts(qc)
        probs  = np.array([c / shots for c in counts.values()], dtype=float)
        return probs

class QuantumFullyConnected:
    """Parameterized circuit that outputs a single expectation value,
    mimicking the classical fully‑connected layer."""
    def __init__(self, n_qubits: int = 1):
        self.n_qubits = n_qubits
        self.backend  = Aer.get_backend("qasm_simulator")
        self.theta = qiskit.circuit.Parameter('theta')

    def _build_circuit(self, theta_val: float) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        qc.barrier()
        qc.ry(_clip(theta_val, 2*np.pi), range(self.n_qubits))
        qc.measure_all()
        return qc

    def run(self,
            theta_vals: Iterable[float],
            shots: int = 1024) -> np.ndarray:
        expectations = []
        for theta in theta_vals:
            qc = self._build_circuit(theta)
            job = execute(qc, self.backend, shots=shots)
            counts = job.result().get_counts(qc)
            probs = np.array([c / shots for c in counts.values()], dtype=float)
            expectation = np.sum(np.array(list(counts.keys()), dtype=float) * probs)
            expectations.append(expectation)
        return np.array(expectations, dtype=float)

class SelfAttentionHybridQuantum:
    """Quantum‑centric self‑attention that fuses attention weights
    and a fully‑connected expectation value."""
    def __init__(self, n_qubits: int):
        self.attention = QuantumSelfAttention(n_qubits)
        self.fc        = QuantumFullyConnected(n_qubits)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            fc_thetas: Iterable[float],
            shots: int = 1024) -> np.ndarray:
        # Obtain attention probability distribution
        attn_weights = self.attention.run(rotation_params, entangle_params, shots)
        # Obtain FC expectations
        fc_expectations = self.fc.run(fc_thetas, shots)
        # Combine by weighted sum (truncate if needed)
        combined = np.sum(attn_weights[:len(fc_expectations)] * fc_expectations)
        return np.array([combined], dtype=float)

__all__ = ["SelfAttentionHybridQuantum"]
