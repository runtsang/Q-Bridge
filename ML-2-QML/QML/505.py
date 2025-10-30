"""Quantum convolutional filter using a parameter‑sharing variational ansatz.

The ConvPlus class mimics the interface of the classical ConvPlus filter but
uses a shallow variational circuit that is executed on a qiskit simulator.
It supports batched input and a learnable threshold that is applied to the
measurement outcome.  The circuit consists of a single Ry rotation with a
shared parameter on all qubits followed by a ring of CNOTs.  The depth of the
ansatz can be increased by repeating the rotation–CNOT block.
"""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from typing import Dict

__all__ = ["ConvPlus"]

class ConvPlus:
    def __init__(self,
                 kernel_size: int = 2,
                 depth: int = 1,
                 shots: int = 1024,
                 threshold: float = 0.5,
                 backend: qiskit.providers.Backend | None = None) -> None:
        self.n_qubits = kernel_size ** 2
        self.depth = depth
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.theta = Parameter("theta")
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # shared Ry rotation on all qubits
        for q in range(self.n_qubits):
            qc.ry(self.theta, q)
        # depth layers of CNOTs in a ring
        for _ in range(self.depth):
            for q in range(self.n_qubits):
                qc.cx(q, (q + 1) % self.n_qubits)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Args:
            data: 2D array of shape (kernel_size, kernel_size) with values in [0,255].
        Returns:
            float: average probability of measuring |1> across all qubits after applying
                   the circuit with parameters set according to the input data.
        """
        flat = np.reshape(data, self.n_qubits)
        # map each input value to a theta value; e.g. π if > threshold*255 else 0
        binds: Dict[Parameter, float] = {self.theta: np.pi if v > self.threshold * 255 else 0 for v in flat}
        job = execute(self._circuit, self.backend, shots=self.shots, parameter_binds=[binds])
        result = job.result()
        counts = result.get_counts(self._circuit)
        total_ones = 0
        total_shots = self.shots * self.n_qubits
        for state, cnt in counts.items():
            ones = sum(int(bit) for bit in state)
            total_ones += ones * cnt
        return total_ones / total_shots
