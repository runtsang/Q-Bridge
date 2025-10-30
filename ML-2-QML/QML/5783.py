from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class QuantumClassifierModel:
    """Hybrid classifier with classical feed‑forward backbone and quantum variational ansatz."""
    def __init__(self, num_features: int, depth: int, use_quantum: bool = True):
        self.num_features = num_features
        self.depth = depth
        self.use_quantum = use_quantum
        self.classical_net, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_features, depth)
        if self.use_quantum:
            self.quantum_circuit, self.quantum_encoding, self.quantum_weights, self.quantum_observables = build_classifier_circuit(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classical network."""
        return self.classical_net(x)

    def forward_quantum(self, x: torch.Tensor) -> torch.Tensor:
        """Quantum forward pass using the variational circuit."""
        if not self.use_quantum:
            raise RuntimeError("Quantum part is disabled.")
        bound_circuit = self.quantum_circuit.bind_parameters({p: v.item() for p, v in zip(self.quantum_encoding, x)})
        job = execute(bound_circuit, Aer.get_backend('qasm_simulator'), shots=1024)
        result = job.result().get_counts(bound_circuit)
        # Simplified expectation evaluation
        expectation = torch.zeros(len(self.quantum_observables))
        for i, _ in enumerate(self.quantum_observables):
            expectation[i] = 0.0
        return expectation

    def forward_hybrid(self, x: torch.Tensor) -> torch.Tensor:
        """Combine classical and quantum outputs."""
        class_out = self.forward(x)
        if self.use_quantum:
            quantum_out = self.forward_quantum(x)
            return torch.cat([class_out, quantum_out], dim=1)
        return class_out
