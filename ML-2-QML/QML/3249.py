"""Unified quantum estimator using Qiskit and FastBaseEstimator logic."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator


class EstimatorQNN:
    """Parameterised quantum circuit with observables and a fast expectation evaluator."""
    def __init__(self) -> None:
        # Parameters
        self.input_params = [Parameter("input1")]
        self.weight_params = [Parameter("weight1")]
        # Circuit
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.rx(self.weight_params[0], 0)
        # Observables
        self.observables: List[BaseOperator] = [
            SparsePauliOp.from_list([("Y", 1)])
        ]

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.input_params) + len(self.weight_params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.input_params + self.weight_params, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator] | None = None,
        parameter_sets: Sequence[Sequence[float]] = (),
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.
        If shots is provided, inject Gaussian noise to emulate measurement statistics."""
        observables = list(observables) if observables is not None else self.observables
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [rng.normal(complex(v.real, v.imag), 1 / shots) for v in row]
            noisy.append(noisy_row)
        return noisy


def EstimatorQNN() -> EstimatorQNN:
    """Return a quantum EstimatorQNN instance ready for evaluation."""
    return EstimatorQNN()


__all__ = ["EstimatorQNN"]
