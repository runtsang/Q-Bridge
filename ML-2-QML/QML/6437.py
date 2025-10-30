"""Quantum hybrid classifier with optional kernel encoding."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class HybridClassifier:
    """
    Quantum circuit that optionally prepends a data‑re‑uploading kernel
    before a variational ansatz.  The class mirrors the classical
    interface so that the same API can be used in hybrid experiments.
    """
    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 use_kernel: bool = False,
                 gamma: float = 1.0):
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_kernel = use_kernel
        self.gamma = gamma
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth, use_kernel, gamma)

    def get_circuit(self) -> QuantumCircuit:
        return self.circuit

def build_classifier_circuit(num_qubits: int,
                             depth: int,
                             use_kernel: bool = False,
                             gamma: float = 1.0) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered ansatz with optional kernel encoding.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)

    # data encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    if use_kernel:
        # simple kernel: additional rotations scaled by gamma
        for i in range(num_qubits):
            qc.ry(gamma * encoding[i], i)

    # variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

__all__ = ["HybridClassifier", "build_classifier_circuit"]
