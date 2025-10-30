"""Enhanced FastBaseEstimator for Qiskit circuits with gradients and shot simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit with gradient support."""

    def __init__(self, circuit: QuantumCircuit, backend: AerSimulator | None = None) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend = backend or AerSimulator()
        self.backend.set_options(method="statevector")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_circuit = self._bind(values)
            state = Statevector.from_instruction(bound_circuit)
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
    ) -> List[List[float]]:
        """Simulate measurements with shot noise."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean.real, max(1e-6, 1 / np.sqrt(shots)))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def gradient(
        self,
        observable: BaseOperator,
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute gradient of a single observable w.r.t parameters using parameter shift rule."""
        gradients: List[List[float]] = []
        shift = np.pi / 2
        for values in parameter_sets:
            grad_row: List[float] = []
            for p_idx in range(len(self.parameters)):
                params_plus = list(values)
                params_minus = list(values)
                params_plus[p_idx] += shift
                params_minus[p_idx] -= shift
                exp_plus = self.evaluate([observable], [params_plus])[0][0].real
                exp_minus = self.evaluate([observable], [params_minus])[0][0].real
                grad = 0.5 * (exp_plus - exp_minus)
                grad_row.append(grad)
            gradients.append(grad_row)
        return gradients


__all__ = ["FastBaseEstimator"]
