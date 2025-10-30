"""Minimal estimator primitive used by the simplified fast primitives package."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit with caching and optional shot noise."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._state_cache: dict[tuple[float], Statevector] = {}

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _get_statevector(self, params: Sequence[float]) -> Statevector:
        key = tuple(params)
        if key not in self._state_cache:
            bound = self._bind(params)
            self._state_cache[key] = Statevector.from_instruction(bound)
        return self._state_cache[key]

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.
        If *shots* is provided, returns a noisy estimate that mimics finite‑shot
        sampling by adding Gaussian noise with variance 1/shots.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        # Exact evaluation
        for params in parameter_sets:
            state = self._get_statevector(params)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        std = 1 / np.sqrt(shots)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                (
                    rng.normal(float(val.real), max(1e-6, std))
                    + 1j * rng.normal(float(val.imag), max(1e-6, std))
                )
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator"]
