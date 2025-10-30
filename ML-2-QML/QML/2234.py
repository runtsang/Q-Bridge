"""Quantum component for the hybrid fraud detection model.

The circuit is a simple variational ansatz inspired by the Qiskit
implementation in the original seed.  It accepts a 2‑dimensional
classical feature vector (the output of the classical network) and
produces a single‑qubit measurement that can be interpreted as a
logit for fraud detection.  The circuit is parameterised by a depth
and an optional number of qubits; the encoding uses RX rotations,
followed by a layered sequence of RY rotations and CZ entanglers.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class HybridFraudQuantum:
    """
    Wrapper around the quantum circuit that can be called from the classical
    hybrid model.  The callable accepts a tensor of shape (batch, 2) and
    returns a tensor of shape (batch, 1) after simulation on a backend.
    """

    def __init__(self, num_qubits: int, depth: int, backend=None):
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)
        if backend is None:
            from qiskit.providers.aer import AerSimulator
            backend = AerSimulator()
        self.backend = backend

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Convert torch tensor to numpy
        import numpy as np
        import torch

        batch = x.shape[0]
        # Prepare parameter bindings
        shots = 1024
        results = []
        for i in range(batch):
            param_dict = {str(p): float(x[i, j]) for j, p in enumerate(self.encoding)}
            # bind variational parameters to zero initially
            for p in self.weights:
                param_dict[str(p)] = 0.0
            bound_circ = self.circuit.bind_parameters(param_dict)
            job = self.backend.run(bound_circ, shots=shots)
            result = job.result()
            counts = result.get_counts()
            # expectation value of Z
            exp = sum((1 if '0' in bit else -1) * count for bit, count in counts.items()) / shots
            results.append(exp)
        return torch.tensor(results, dtype=torch.float32).unsqueeze(-1)

__all__ = ["build_classifier_circuit", "HybridFraudQuantum"]
