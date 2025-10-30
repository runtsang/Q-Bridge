import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute

class HybridFCL:
    """
    Quantum layer exposing a 2‑qubit variational circuit.
    The circuit consists of a Ry gate on each qubit, a CX entanglement,
    and measurement of the first qubit in the Pauli‑Z basis to produce
    an expectation value that serves as a quantum feature.
    """
    def __init__(self, n_qubits: int = 2, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        for q in range(self.n_qubits):
            qc.ry(self.theta, q)
        qc.cx(0, 1)
        qc.measure_all()
        return qc

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result()
        counts = result.get_counts(self._circuit)
        exp = 0.0
        for bitstring, cnt in counts.items():
            prob = cnt / self.shots
            bit = int(bitstring[0])
            exp += (1 if bit == 0 else -1) * prob
        return np.array([exp])
