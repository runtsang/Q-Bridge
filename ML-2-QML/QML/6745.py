"""Hybrid estimator for Qiskit quantum circuits with shot‑noise simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator
import numpy as np


class HybridBaseEstimator:
    """Evaluate a parametrized Qiskit circuit and optionally simulate shot noise."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("HybridBaseEstimator expects a qiskit.QuantumCircuit.")
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._simulator = AerSimulator(method="statevector")

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
        """Compute expectation values for each parameter set."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Return expectation values sampled from Aer with optional shots."""
        if shots is None:
            return self.evaluate(observables, parameter_sets)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            job = self._simulator.run(bound, shots=shots, seed_simulator=seed)
            result = job.result()
            exp_vals = []
            for obs in observables:
                exp = result.get_expectation_value(obs, bound)
                exp_vals.append(exp)
            results.append(exp_vals)
        return results


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, Iterable, Iterable, list[BaseOperator]]:
    """Construct a layered ansatz with encoding, variational parameters, and observables."""
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

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


__all__ = ["HybridBaseEstimator", "build_classifier_circuit"]
