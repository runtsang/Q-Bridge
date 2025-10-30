"""Quantum-aware estimator with variational circuits and quantum transformer blocks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class QuantumTransformerBlock:
    """A single quantum transformer block consisting of parametric rotations and entangling gates."""
    def __init__(self, n_qubits: int, block_id: int):
        self.n_qubits = n_qubits
        self.circuit = QuantumCircuit(n_qubits)
        self.params: List[Parameter] = []

        # First layer of RX rotations
        for i in range(n_qubits):
            p = Parameter(f"block{block_id}_rx{i}")
            self.params.append(p)
            self.circuit.rx(p, i)

        # Entangling CNOT chain (acts like a simplified attention)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.cx(n_qubits - 1, 0)  # wrap‑around for cyclic entanglement

        # Second layer of RY rotations
        for i in range(n_qubits):
            p = Parameter(f"block{block_id}_ry{i}")
            self.params.append(p)
            self.circuit.ry(p, i)


class QuantumTextTransformer:
    """Stack of quantum transformer blocks that can be bound to a set of parameters."""
    def __init__(self, n_qubits: int, num_blocks: int):
        self.n_qubits = n_qubits
        self.blocks = [QuantumTransformerBlock(n_qubits, i) for i in range(num_blocks)]
        self.circuit = QuantumCircuit(n_qubits)
        for block in self.blocks:
            self.circuit.append(block.circuit, range(n_qubits))
        self.parameters = [p for block in self.blocks for p in block.params]

    def bind_parameters(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.parameters, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)


class UnifiedEstimatorTransformer(FastBaseEstimator):
    """Hybrid quantum estimator that evaluates a quantum transformer circuit."""
    def __init__(self, transformer: QuantumTextTransformer):
        super().__init__(transformer.circuit)
        self.transformer = transformer

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound_circ = self.transformer.bind_parameters(params)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


__all__ = [
    "FastBaseEstimator",
    "QuantumTransformerBlock",
    "QuantumTextTransformer",
    "UnifiedEstimatorTransformer",
]
