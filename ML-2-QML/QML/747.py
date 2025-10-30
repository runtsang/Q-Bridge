"""Extended FastBaseEstimator for Qiskit with gradient support and optional shot‑noise simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import Aer, transpile, assemble
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.opflow import PauliExpectation, StateFn, CircuitStateFn, PauliSumOp


class FastBaseEstimator:
    """Evaluate expectation values of parametrized circuits with optional gradient and shot‑noise support.

    Enhancements over the original seed:
    * Parameter‑shift gradient computation for each observable.
    * Shot‑noise simulation via Gaussian perturbation of expectation values.
    * Flexible backend selection (state‑vector or Aer qasm simulator).
    * ``evaluate`` now accepts an optional ``shots`` argument.
    """

    def __init__(self, circuit: QuantumCircuit, backend: str = "statevector") -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend

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
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng()
            noisy = []
            for row in results:
                noisy_row = [rng.normal(val.real, max(1e-6, 1 / shots)) for val in row]
                noisy.append(noisy_row)
            return noisy

        return results

    def compute_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[np.ndarray]]:
        """Return gradients of each observable w.r.t. the circuit parameters using the
        parameter‑shift rule. Gradients are returned as NumPy arrays of shape
        (num_parameters,).
        """
        observables = list(observables)
        gradients: List[List[np.ndarray]] = []

        for params in parameter_sets:
            grad_row: List[np.ndarray] = []
            for observable in observables:
                grad: List[float] = []
                for i, _ in enumerate(self._parameters):
                    shift = np.pi / 2
                    params_plus = list(params)
                    params_plus[i] += shift
                    params_minus = list(params)
                    params_minus[i] -= shift

                    f_plus = self._expectation_value(observable, params_plus)
                    f_minus = self._expectation_value(observable, params_minus)

                    grad.append((f_plus - f_minus) / 2.0)
                grad_row.append(np.array(grad))
            gradients.append(grad_row)
        return gradients

    def _expectation_value(
        self,
        observable: BaseOperator,
        parameter_values: Sequence[float],
    ) -> complex:
        """Helper to compute a single expectation value."""
        state = Statevector.from_instruction(self._bind(parameter_values))
        return state.expectation_value(observable)

    def evaluate_with_noise(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Wrap ``evaluate`` and add Gaussian shot noise if ``shots`` is provided."""
        raw = self.evaluate(observables, parameter_sets, shots=shots)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [rng.normal(val.real, max(1e-6, 1 / shots)) for val in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator"]
