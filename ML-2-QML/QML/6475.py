"""
Quantum implementation of the UnifiedClassifier.
Builds a layered ansatz with data encoding, variational gates, and Pauli‑Z observables.
"""

from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class UnifiedClassifier:
    """
    Quantum variational classifier with a unified `run` interface.
    """
    def __init__(self, num_qubits: int, depth: int = 2, backend=None, shots: int = 1024) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, list, list, list[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        circuit = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            circuit.rx(param, qubit)
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        observables = [SparsePauliOp("I"*i + "Z" + "I"*(self.num_qubits-i-1)) for i in range(self.num_qubits)]
        return circuit, list(encoding), list(weights), observables

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the variational circuit with the supplied parameters and return
        the expectation values of the Pauli‑Z observables.
        """
        if len(thetas)!= len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} parameters, got {len(thetas)}.")
        param_bind = [{w: t for w, t in zip(self.weights, thetas)}]
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_bind)
        result = job.result()
        expectation = np.array([result.get_expectation_value(op, self.circuit) for op in self.observables])
        return expectation

__all__ = ["UnifiedClassifier"]
