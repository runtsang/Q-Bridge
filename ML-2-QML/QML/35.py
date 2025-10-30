"""HybridClassifier: quantum variational classifier with data re‑uploading.

This module mirrors the classical implementation but uses a parameterised
ansatz built with Qiskit.  The class exposes the same metadata interface
(`encoding`, `weights`, `observables`) so that the two flavours can be
interchanged in hybrid training loops.

Key extensions:
- Data‑re‑uploading ansatz with configurable depth and qubit count.
- Support for arbitrary Pauli‑Z observables per qubit.
- Utility to convert classical weights to quantum parameters.
"""

from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class HybridClassifier:
    def __init__(self,
                 num_qubits: int,
                 depth: int = 3) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = self.build_circuit()

    def build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        # data encoding
        for qubit, param in enumerate(encoding):
            qc.rx(param, qubit)

        # variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # measurement observables
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                       for i in range(self.num_qubits)]
        return qc, [encoding], [weights], observables

    def metadata(self) -> Tuple[List[int], List[int], List[int]]:
        """Return encoding indices, weight sizes and observable indices."""
        encoding = list(range(self.num_qubits))
        weight_sizes = [self.num_qubits * self.depth]
        observables = list(range(self.num_qubits))
        return encoding, weight_sizes, observables

    def bind_parameters(self, data: List[float], params: List[float]) -> QuantumCircuit:
        """Return a circuit with data and variational parameters bound."""
        bound = self.circuit.bind_parameters(
            {self.encoding[0][i]: data[i] for i in range(self.num_qubits)}
            | {self.weights[0][i]: params[i] for i in range(len(params))}
        )
        return bound

__all__ = ["HybridClassifier"]
