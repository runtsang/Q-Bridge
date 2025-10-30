import numpy as np
import qiskit
from qiskit import QuantumCircuit as _QC, transpile, assemble, Aer

class HybridQuantumFullyConnectedLayer:
    """Quantum circuit wrapper for a fully connected layer."""
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 100):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        qc = _QC(self.n_qubits)
        all_qubits = list(range(self.n_qubits))
        theta = qiskit.circuit.Parameter("theta")
        qc.h(all_qubits)
        qc.barrier()
        qc.ry(theta, all_qubits)
        qc.measure_all()
        return qc

    def run(self, angles: np.ndarray) -> np.ndarray:
        if not isinstance(angles, np.ndarray):
            angles = np.array(angles)
        if angles.ndim == 0:
            angles = angles.reshape(1)
        compiled = transpile(self.circuit, self.backend)
        expectations = []
        for ang in angles:
            param_binds = [{self.circuit.parameters[0]: ang}]
            qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
            probs = np.array(list(counts.values())) / self.shots
            expectation = np.sum(states * probs)
            expectations.append(expectation)
        return np.array(expectations)
