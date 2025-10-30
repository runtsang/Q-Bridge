"""Quantum hybrid filter that simulates a random quantum circuit on 2x2 image patches."""

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

__all__ = ["ConvGen025"]

class ConvGen025:
    """
    Quantum filter that applies a random 2-qubit circuit to a 2x2 image patch.

    The filter can be used as a drop-in replacement for the classical Conv() function.
    It returns a probability-like score for the input patch.
    """
    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100,
                 threshold: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        self.circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(self.n_qubits, 2)
        qc.measure_all()
        return qc

    def run(self, data) -> float:
        """
        Run the quantum filter on a 2D array and return a probability score.

        Args:
            data: 2D array of shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1> across qubits.
        """
        # Flatten data
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data_flat:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.circuit.parameters[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(self.circuit, self.backend, shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)
        total = 0
        for key, val in counts.items():
            ones = sum(int(bit) for bit in key)
            total += ones * val
        return total / (self.shots * self.n_qubits)
