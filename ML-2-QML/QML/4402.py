from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[qiskit.QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = qiskit.QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class QuantumCircuitWrapper:
    """Thin wrapper that executes a Qiskit circuit and returns expectation values of Z on the first qubit."""

    def __init__(self, circuit: qiskit.QuantumCircuit, backend, shots: int = 1024):
        self.circuit = circuit
        self.backend = backend
        self.shots = shots
        self.param_names = list(circuit.parameters)

    def run(self, params: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{p: val} for p, val in zip(self.param_names, params)],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(counts):
            exp = 0.0
            for state, count in counts.items():
                z = 1.0 if state[0] == "1" else -1.0
                exp += z * count
            return exp / self.shots

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])
