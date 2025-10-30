import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumConvCircuit:
    """Quantum circuit that emulates a convolutional filter."""
    def __init__(self,
                 kernel_size: int = 2,
                 backend=None,
                 shots: int = 100):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or AerSimulator()
        self.shots = shots
        self._build_circuit()

    def _build_circuit(self):
        self.circuit = QuantumCircuit(self.n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        self.theta = theta
        self.circuit.h(range(self.n_qubits))
        for i in range(self.n_qubits):
            self.circuit.rx(theta[i], i)
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()
        self.compiled = transpile(self.circuit, self.backend)

    def run(self, data: np.ndarray) -> float:
        flat = data.flatten()
        param_binds = [{self.theta[i]: np.pi if val > 0 else 0}
                       for i, val in enumerate(flat)]
        qobj = assemble(self.compiled, shots=self.shots,
                        parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts(self.circuit)
        exp_z = 0.0
        for bitstr, cnt in counts.items():
            z = 1 if bitstr[-1] == '1' else -1
            exp_z += z * cnt
        exp_z /= self.shots
        return exp_z

def Conv(kernel_size: int = 2, shots: int = 100):
    return QuantumConvCircuit(kernel_size=kernel_size, shots=shots)
