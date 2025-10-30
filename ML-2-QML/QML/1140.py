"""
Quantum convolution filter module.
Provides a parameterized quantum circuit that can be used as a drop‑in
replacement for a classical Conv layer.  The class exposes a `run`
method that accepts a 2‑D array and returns a scalar output, and a
`gradient` method that computes gradients w.r.t. the rotation angles.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit import execute

class ConvEnhanced:
    """
    Quantum convolution filter that maps a 2D kernel to a scalar output.
    The filter uses a parameterized rotation feature map and a simple
    entangling layer.  Parameters are stored as qiskit.Parameters and
    can be updated externally.
    """
    def __init__(self, kernel_size: int = 2, shots: int = 1024, threshold: float = 0.0):
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        self.circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.ry(self.theta[i], i)
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()
        self.backend = AerSimulator()

    def run(self, data: np.ndarray) -> float:
        data_flat = np.reshape(data, (self.n_qubits,))
        angles = np.where(data_flat > self.threshold, np.pi, 0.0)
        bind_dict = {self.theta[i]: angles[i] for i in range(self.n_qubits)}
        bound_circuit = self.circuit.bind_parameters(bind_dict)
        job = execute(bound_circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)
        total_ones = sum(bitstring.count('1') * count for bitstring, count in counts.items())
        return total_ones / (self.shots * self.n_qubits)

    def set_parameters(self, params: np.ndarray):
        for i, val in enumerate(params):
            self.theta[i].set_value(val)

    def get_parameters(self) -> np.ndarray:
        return np.array([self.theta[i].value() for i in range(self.n_qubits)])

    def gradient(self, data: np.ndarray) -> np.ndarray:
        shift = np.pi / 2
        grad = np.zeros(self.n_qubits)
        base_angles = np.where(data.flatten() > self.threshold, np.pi, 0.0)
        for i in range(self.n_qubits):
            plus = base_angles.copy()
            minus = base_angles.copy()
            plus[i] += shift
            minus[i] -= shift
            bind_plus = {self.theta[j]: plus[j] for j in range(self.n_qubits)}
            circ_plus = self.circuit.bind_parameters(bind_plus)
            job_plus = execute(circ_plus, self.backend, shots=self.shots)
            res_plus = job_plus.result().get_counts(circ_plus)
            ones_plus = sum(bitstring.count('1') * c for bitstring, c in res_plus.items())
            prob_plus = ones_plus / (self.shots * self.n_qubits)
            bind_minus = {self.theta[j]: minus[j] for j in range(self.n_qubits)}
            circ_minus = self.circuit.bind_parameters(bind_minus)
            job_minus = execute(circ_minus, self.backend, shots=self.shots)
            res_minus = job_minus.result().get_counts(circ_minus)
            ones_minus = sum(bitstring.count('1') * c for bitstring, c in res_minus.items())
            prob_minus = ones_minus / (self.shots * self.n_qubits)
            grad[i] = (prob_plus - prob_minus) / (2 * shift)
        return grad

__all__ = ["ConvEnhanced"]
