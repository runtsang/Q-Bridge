import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import Aer, execute

class ConvEnhanced:
    """Quantum filter for quanvolution layers."""
    def __init__(self, kernel_size, backend=None, shots=1024, threshold=127):
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuit = self._build_circuit(kernel_size)

    def _build_circuit(self, kernel_size):
        qc = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            qc.rx(self.theta[i], i)
        qc.barrier()
        qc += random_circuit(self.n_qubits, 2, seed=42)
        qc.measure_all()
        return qc

    def run(self, data):
        """Run the quantum circuit on classical data."""
        flat = np.reshape(data, (self.n_qubits))
        param_binds = {}
        for i, val in enumerate(flat):
            param_binds[self.theta[i]] = np.pi if val > self.threshold else 0
        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[param_binds])
        result = job.result()
        counts = result.get_counts(self.circuit)
        total_ones = 0
        for key, val in counts.items():
            total_ones += sum(int(bit) for bit in key) * val
        return total_ones / (self.shots * self.n_qubits)
