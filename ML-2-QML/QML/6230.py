import numpy as np
import qiskit
from qiskit import QuantumCircuit as QC, transpile, assemble
from qiskit.providers.aer import AerSimulator

class UnifiedHybridLayer:
    """
    Parameterised quantum circuit that evaluates the expectation value
    of the Z operator on the first qubit after a Ry rotation. The class
    can be reused by the classical hybrid layer via its ``run`` method.
    """
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend if backend is not None else AerSimulator()
        self._circuit_template = QC(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit_template.h(range(n_qubits))
        self._circuit_template.ry(self.theta, range(n_qubits))
        self._circuit_template.measure_all()
        self._compiled = transpile(self._circuit_template, self.backend)

    def run(self, angles: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for each angle in ``angles`` and return a
        NumPy array of expectation values. The expectation is computed
        as the average of Z on the first qubit.
        """
        expectations = []
        for a in angles:
            bind = {self.theta: float(a)}
            qobj = assemble(self._compiled,
                            shots=self.shots,
                            parameter_binds=[bind])
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts(self._compiled)
            exp = 0.0
            for bitstring, cnt in counts.items():
                z = 1 if bitstring[-1] == '0' else -1
                exp += z * cnt
            exp /= self.shots
            expectations.append(exp)
        return np.array(expectations)

__all__ = ["UnifiedHybridLayer"]
