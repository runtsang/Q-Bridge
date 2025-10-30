"""Hybrid estimator using Qiskit with configurable shot noise and parameter‑shift gradients."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class HybridEstimator:
    """Evaluate expectation values of observables for a parametrized circuit.

    Supports optional shot‑noise simulation and a helper for computing
    parameter‑shift gradients.
    """

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
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Quantum operators whose expectation values are desired.
        parameter_sets : sequence of parameter sequences
            Each inner sequence is bound to the circuit.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added to each result.
        seed : int, optional
            Random seed for reproducibility of the shot noise.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noise_std = max(1e-6, 1.0 / shots)
            noisy_results = [
                [rng.normal(np.real(val), noise_std) + 1j * rng.normal(np.imag(val), noise_std)
                 for val in row]
                for row in results
            ]
            return noisy_results
        return results

    def parameter_shift_gradient(
        self,
        observable: BaseOperator,
        parameter_index: int,
        shift: float = np.pi / 2,
    ) -> Callable[[Sequence[float]], float]:
        """Return a function that computes the gradient of the expectation value
        of ``observable`` with respect to the parameter at ``parameter_index`` using
        the parameter‑shift rule."""
        def grad_func(params: Sequence[float]) -> float:
            params = list(params)
            plus = params.copy()
            minus = params.copy()
            plus[parameter_index] += shift
            minus[parameter_index] -= shift
            plus_state = Statevector.from_instruction(self._bind(plus))
            minus_state = Statevector.from_instruction(self._bind(minus))
            plus_exp = plus_state.expectation_value(observable)
            minus_exp = minus_state.expectation_value(observable)
            return (plus_exp - minus_exp).real / (2 * np.sin(shift))
        return grad_func


__all__ = ["HybridEstimator"]
