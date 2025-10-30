"""Quantum classifier with parameterized ansatz and expectation evaluation."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumClassifierModel:
    """Parameterized quantum circuit classifier with data‑encoding and variational layers."""

    def __init__(self, num_qubits: int, depth: int = 3) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        for qubit, param in enumerate(encoding):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1)) for i in range(self.num_qubits)]
        return qc, [encoding], [weights], observables

    def evaluate(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Return expectation values for each qubit using the Aer simulator."""
        if data.shape[0]!= self.num_qubits:
            raise ValueError("Data dimension must match number of qubits.")

        bound_circuit = self.circuit
        bound_circuit = bound_circuit.bind_parameters({p: val for p, val in zip(self.encoding[0], data)})
        bound_circuit = bound_circuit.bind_parameters({p: val for p, val in zip(self.weights[0], params)})

        backend = Aer.get_backend("statevector_simulator")
        result = execute(bound_circuit, backend, shots=1024).result()
        statevector = result.get_statevector(bound_circuit)
        exp_vals = np.array([np.real(np.dot(statevector.conj(), op.to_matrix().dot(statevector))) for op in self.observables])
        return exp_vals

    def predict(self, data: np.ndarray) -> int:
        """Predict class label based on sign of expectation values."""
        exp_vals = self.evaluate(params=np.zeros(self.weights[0].size), data=data)
        return int(exp_vals.mean() > 0)

    def get_metadata(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        return self.circuit, self.encoding, self.weights, self.observables


def build_classifier_circuit(num_qubits: int, depth: int = 3) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
    model = QuantumClassifierModel(num_qubits, depth)
    return model.circuit, model.encoding, model.weights, model.observables


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
