from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import List, Tuple

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class FraudDetectionHybrid:
    """
    Quantum counterpart of FraudDetectionHybrid.
    Builds a data‑encoding and variational circuit with Pauli‑Z observables.
    """
    def __init__(self, num_qubits: int = 4, depth: int = 3) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = self.build_circuit()

    def build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        qc = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1)) for i in range(self.num_qubits)]
        return qc, list(encoding), list(weights), observables

    def parameter_sizes(self) -> List[int]:
        return [self.num_qubits] + [self.num_qubits * self.depth]

    def get_circuit(self) -> QuantumCircuit:
        return self.circuit

__all__ = ["FraudDetectionHybrid"]
