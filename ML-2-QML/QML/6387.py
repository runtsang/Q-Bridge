"""Hybrid quantum self‑attention with a fully‑connected quantum layer.

The SelfAttention() factory returns a QuantumHybridSelfAttention instance
that builds a parameterized circuit consisting of per‑qubit rotations,
controlled‑X entanglement, and a single‑qubit Ry fully‑connected layer.
The run method returns the expectation value of the measured computational
basis states.  The implementation uses qiskit’s Aer simulator but can be
traded for any Qiskit backend."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def SelfAttention():
    class QuantumHybridSelfAttention:
        """Quantum self‑attention with a fully‑connected layer."""

        def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
            self.n_qubits = n_qubits
            self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
            self.shots = shots

        def _build_circuit(
            self, rotation_params: np.ndarray, entangle_params: np.ndarray, fcl_theta: float
        ) -> QuantumCircuit:
            qc = QuantumCircuit(self.n_qubits)
            # Per‑qubit rotations
            for i in range(self.n_qubits):
                qc.rx(rotation_params[3 * i], i)
                qc.ry(rotation_params[3 * i + 1], i)
                qc.rz(rotation_params[3 * i + 2], i)

            # Entanglement via controlled‑X rotations
            for i in range(self.n_qubits - 1):
                qc.crx(entangle_params[i], i, i + 1)

            # Fully‑connected quantum layer (single Ry per qubit)
            qc.h(range(self.n_qubits))
            qc.barrier()
            qc.ry(fcl_theta, range(self.n_qubits))

            # Measurement
            qc.measure_all()
            return qc

        def run(
            self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            fcl_theta: float,
        ) -> np.ndarray:
            qc = self._build_circuit(rotation_params, entangle_params, fcl_theta)
            job = qiskit.execute(qc, self.backend, shots=self.shots)
            result = job.result().get_counts(qc)
            counts = np.array(list(result.values()))
            states = np.array([int(state, 2) for state in result.keys()])
            probs = counts / self.shots
            expectation = np.sum(states * probs)
            return np.array([expectation])

    return QuantumHybridSelfAttention()
