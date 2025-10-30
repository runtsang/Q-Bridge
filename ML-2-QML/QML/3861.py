from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator, SparsePauliOp
from collections.abc import Iterable, Sequence
from typing import List

class FastBaseEstimator:
    """Fast evaluation for parametrized Qiskit circuits with optional shot‑noise emulation."""

    def __init__(self, circuit: QuantumCircuit) -> None:
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
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = []
            for val in row:
                noise = rng.normal(0, 1 / shots)
                noisy_val = complex(val.real + noise, val.imag)
                noisy_row.append(noisy_val)
            noisy.append(noisy_row)
        return noisy


def EstimatorQNN() -> tuple["FastBaseEstimator", List[BaseOperator]]:
    """Return a toy quantum circuit and its observable used in the EstimatorQNN example."""
    from qiskit.circuit import Parameter
    params = [Parameter("input1"), Parameter("weight1")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)

    observable = SparsePauliOp.from_list([("Y", 1)])
    return FastBaseEstimator(qc), [observable]


__all__ = ["FastBaseEstimator", "EstimatorQNN"]
