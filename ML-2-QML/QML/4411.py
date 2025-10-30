import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class QuantumCircuit:
    """Parametrised two‑qubit circuit that returns the expectation of Z on the first qubit."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 100):
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> float:
        """Execute the circuit for a vector of angles and return the Z‑expectation."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: p} for p in params],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        exp = 0.0
        for state, cnt in result.items():
            bit = int(state[-1])  # measurement order: last bit is first qubit
            prob = cnt / self.shots
            exp += (1 - 2 * bit) * prob
        return exp

def get_quantum_callable(n_qubits: int = 2, shots: int = 100):
    """Return a callable that forwards a numpy array of angles to the quantum circuit."""
    qc = QuantumCircuit(n_qubits, shots=shots)
    return lambda params: qc.run(params)

__all__ = ["QuantumCircuit", "get_quantum_callable"]
