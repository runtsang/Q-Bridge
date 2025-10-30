"""Hybrid quantum estimator that mirrors the classical architecture.

The quantum circuit employs a data‑encoding layer followed by a
depth‑dependent variational ansatz.  The observable is chosen to
match the task: a single Y operator for regression or two Z
operators for binary classification.  The estimator is wrapped in
qiskit_machine_learning.neural_networks.EstimatorQNN, providing
the same training API as the classical model.
"""

from __future__ import annotations

from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit.primitives import StatevectorEstimator

class HybridEstimatorQNNQuantum:
    def __init__(self, num_qubits: int = 2, depth: int = 1, task: str = "regression") -> None:
        self.task = task
        self.num_qubits = num_qubits
        self.depth = depth

        # Parameter vectors
        self.encoding_params = ParameterVector("x", num_qubits)
        self.weight_params = ParameterVector("theta", num_qubits * depth)

        # Build circuit
        self.circuit = QuantumCircuit(num_qubits)
        for param, qubit in zip(self.encoding_params, range(num_qubits)):
            self.circuit.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                self.circuit.ry(self.weight_params[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                self.circuit.cz(qubit, qubit + 1)

        # Observables
        if task == "regression":
            self.observables = [SparsePauliOp.from_list([("Y" * num_qubits, 1)])]
        else:  # classification
            self.observables = [
                SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])] * 2

        # Estimator wrapper
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = QEstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.encoding_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def get_estimator(self):
        """Return the wrapped qiskit EstimatorQNN instance."""
        return self.estimator_qnn

def EstimatorQNN(num_qubits: int = 2, depth: int = 1, task: str = "regression") -> HybridEstimatorQNNQuantum:
    """Return a hybrid quantum estimator instance with default configuration."""
    return HybridEstimatorQNNQuantum(num_qubits, depth, task)

__all__ = ["EstimatorQNN", "HybridEstimatorQNNQuantum"]
