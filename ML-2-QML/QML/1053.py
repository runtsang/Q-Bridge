"""ConvGen351: a quantum filter using a parameter‑shared variational circuit.

This module implements a drop‑in quantum replacement for the original Conv filter.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

class ConvGen351:
    """Quantum variant of ConvGen351.

    The circuit uses n_qubits = kernel_size ** 2 qubits, with a single Ry rotation per qubit
    parameterized by a threshold.  The output is the average probability of measuring |1>
    across all qubits.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int = 100,
        threshold: float = 0.0,
    ):
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        qc = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            qc.ry(self.theta[i], i)
        qc.h(range(self.n_qubits))
        qc.barrier()
        qc.measure_all()
        return qc

    def run(self, data) -> float:
        """Run the quantum filter on 2‑D data.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1> across qubits.
        """
        flat = np.reshape(data, -1)
        binds = [{self.theta[i]: np.pi if val > self.threshold else 0.0 for i, val in enumerate(flat)}]
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=binds)
        result = job.result()
        counts = result.get_counts(self.circuit)
        total = 0
        for bitstring, count in counts.items():
            ones = bitstring.count("1")
            total += ones * count
        return total / (self.shots * self.n_qubits)
