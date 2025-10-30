"""Enhanced quantum estimator for parametrized circuits using Qiskit."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit import Aer
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List, Optional


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""

    def __init__(self, circuit: QuantumCircuit, backend: Optional[str] = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend or Aer.get_backend("statevector_simulator")

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
        enable_error_mitigation: bool = False,
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.

        If ``shots`` is ``None`` the state‑vector simulator is used for exact
        expectation values.  When ``shots`` is provided the same exact values
        are returned but the caller can later add shot‑noise perturbations
        via :class:`FastEstimator`.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            if enable_error_mitigation:
                row = [self._mitigate(val) for val in row]
            results.append(row)

        return results

    @staticmethod
    def _mitigate(value: complex) -> complex:
        """Placeholder for a measurement‑error mitigation routine."""
        return value


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic quantum estimator."""

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        enable_error_mitigation: bool = False,
    ) -> List[List[complex]]:
        raw = super().evaluate(
            observables,
            parameter_sets,
            shots=None,
            enable_error_mitigation=enable_error_mitigation,
        )
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                complex(
                    rng.normal(val.real, max(1e-6, 1 / shots)),
                    rng.normal(val.imag, max(1e-6, 1 / shots)),
                )
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
