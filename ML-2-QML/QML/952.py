"""Hybrid estimator that unifies classical neural network and quantum circuit evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class HybridFastEstimator:
    """Evaluate expectation values of observables for a parametrised quantum circuit.

    The estimator supports parameter‑shift gradient estimation and optional shot
    noise injection.  It is compatible with the classical counterpart via the
    shared class name.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        """Create an evaluator for the given parametrised circuit.

        Parameters
        ----------
        circuit:
            A ``QuantumCircuit`` that may contain parameters.
        """
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a copy of the circuit with parameters bound to the supplied values."""
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Add shot noise to the deterministic expectation values."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                rng.normal(val.real, max(1e-6, 1 / shots)) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

    def evaluate_gradient(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shift: float = np.pi / 2,
    ) -> List[List[List[complex]]]:
        """Compute gradients of each observable w.r.t each parameter using the
        parameter‑shift rule.

        Returns a list of shape (num_sets, num_observables, num_params).
        """
        observables = list(observables)
        gradients: List[List[List[complex]]] = []
        for values in parameter_sets:
            grad_row: List[List[complex]] = []
            for obs in observables:
                grad_params: List[complex] = []
                for idx, _ in enumerate(self._parameters):
                    shift_vec = np.array(values, copy=True)
                    shift_vec[idx] += shift
                    state_plus = Statevector.from_instruction(self._bind(shift_vec))
                    exp_plus = state_plus.expectation_value(obs)

                    shift_vec[idx] -= 2 * shift
                    state_minus = Statevector.from_instruction(self._bind(shift_vec))
                    exp_minus = state_minus.expectation_value(obs)

                    grad = 0.5 * (exp_plus - exp_minus) / np.sin(shift)
                    grad_params.append(grad)
                grad_row.append(grad_params)
            gradients.append(grad_row)
        return gradients


__all__ = ["HybridFastEstimator"]
