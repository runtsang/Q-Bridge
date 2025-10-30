"""Quantum sampler network with classical evaluation interface."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN

class QuantumSamplerNetwork:
    """Hybrid quantum sampler with sampling and expectation utilities."""

    def __init__(self, num_qubits: int = 2, depth: int = 1) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.inputs = ParameterVector("x", self.num_qubits)
        self.weights = ParameterVector("theta", self.num_qubits * depth)
        self.circuit = self._build_circuit()
        self.sampler = Sampler()
        self.qnn = QiskitSamplerQNN(
            circuit=self.circuit,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=self.sampler
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        for qubit, param in zip(range(self.num_qubits), self.inputs):
            qc.rx(param, qubit)
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        return qc

    def sample(self, input_vals: list[float], weight_vals: list[float]) -> np.ndarray:
        bind = dict(zip(self.inputs, input_vals))
        bind.update(zip(self.weights, weight_vals))
        result = self.sampler.run(self.circuit, parameter_binds=[bind])
        return result.quasi_distribution()

    def expectation(
        self,
        observables: list[SparsePauliOp],
        input_vals: list[float],
        weight_vals: list[float],
    ) -> list[complex]:
        bind = dict(zip(self.inputs, input_vals))
        bind.update(zip(self.weights, weight_vals))
        return self.qnn.evaluate(observables, bind)

class FastQuantumEstimator:
    """Fast estimator that wraps the quantum sampler for expectation evaluation."""

    def __init__(self, qnn: QuantumSamplerNetwork) -> None:
        self.qnn = qnn

    def evaluate(
        self,
        observables: list[SparsePauliOp],
        parameter_sets: list[tuple[list[float], list[float]]],
    ) -> list[list[complex]]:
        results: list[list[complex]] = []
        for inp, wts in parameter_sets:
            res = self.qnn.expectation(observables, inp, wts)
            results.append(res)
        return results


__all__ = ["QuantumSamplerNetwork", "FastQuantumEstimator"]
