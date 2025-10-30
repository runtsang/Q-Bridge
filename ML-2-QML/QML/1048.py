"""FastBaseEstimator for quantum circuits with statevector evaluation, shot‑noise emulation, and parameter‑shift gradients."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit.

    Enhancements
    ------------
    * Optional Gaussian shot noise to emulate measurement statistics.
    * Gradient computation via the parameter‑shift rule.
    * Caching of bound circuits for repeated evaluations.
    """

    def __init__(self, circuit: QuantumCircuit, backend: str | None = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = (
            Aer.get_backend(backend) if backend is not None else Aer.get_backend("aer_simulator_statevector")
        )
        self._cache: dict[tuple[float,...], Statevector] = {}

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _cached_statevector(self, values: Sequence[float]) -> Statevector:
        key = tuple(values)
        if key not in self._cache:
            circ = self._bind(values)
            self._cache[key] = Statevector.from_instruction(circ)
        return self._cache[key]

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Observables to evaluate.
        parameter_sets : sequence of parameter sequences
            Each inner sequence contains the values for the circuit parameters.
        shots : int or None, optional
            When provided, Gaussian noise with variance ``1/shots`` is added to
            each expectation value to emulate finite‑sample statistics.
        seed : int or None, optional
            Random seed for reproducibility of shot noise.

        Returns
        -------
        List[List[complex]]
            Outer list indexed by parameter set, inner list indexed by observable.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        rng = np.random.default_rng(seed) if shots is not None else None

        for values in parameter_sets:
            state = self._cached_statevector(values)
            row: List[complex] = [state.expectation_value(obs) for obs in observables]
            if shots is not None:
                row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            results.append(row)
        return results

    def gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[float]]]:
        """Compute gradients of each observable w.r.t. the circuit parameters
        using the parameter‑shift rule.

        Returns
        -------
        List[List[List[float]]]
            Outer list over parameter sets,
            next over observables,
            innermost over parameters.
        """
        observables = list(observables)
        grads: List[List[List[float]]] = []

        shift = np.pi / 2

        for values in parameter_sets:
            grad_row: List[List[float]] = []

            for obs in observables:
                grad_obs: List[float] = []

                for i in range(len(self._parameters)):
                    values_plus = list(values)
                    values_minus = list(values)
                    values_plus[i] += shift
                    values_minus[i] -= shift

                    state_plus = self._cached_statevector(values_plus)
                    state_minus = self._cached_statevector(values_minus)

                    exp_plus = state_plus.expectation_value(obs)
                    exp_minus = state_minus.expectation_value(obs)

                    grad = (exp_plus - exp_minus) / 2
                    grad_obs.append(float(grad))

                grad_row.append(grad_obs)

            grads.append(grad_row)

        return grads


__all__ = ["FastBaseEstimator"]
