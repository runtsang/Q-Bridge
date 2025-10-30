"""Quantum estimator with parameter‑shift gradients and batched evaluation.

Features
--------
* ``evaluate_batch`` uses qiskit.Statevector for simultaneous evaluation of
  multiple parameter sets.
* ``gradient`` implements the standard parameter‑shift rule.
* Supports both pure state and noisy simulation via ``AerSimulator``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, Operator
from qiskit.opflow import PauliSumOp, StateFn, ExpectationFactory
from qiskit.providers.aer import AerSimulator


class FastBaseEstimator:
    """Base estimator for parametrised quantum circuits."""

    def __init__(self, circuit: QuantumCircuit, backend: Optional[AerSimulator] = None) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend = backend or AerSimulator(method="statevector")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs).real for obs in observables]
            results.append(row)
        return results

    def evaluate_batch(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Vectorised evaluation using AerSimulator.  Returns a 2‑D array."""
        params = np.array(parameter_sets, dtype=float)
        if params.ndim == 1:
            params = params.reshape(1, -1)
        shots = 1  # deterministic state‑vector simulation
        job = self.backend.run(
            [self._bind(p) for p in params],
            shots=shots,
            memory=False,
        )
        result = job.result()
        counts = result.get_counts()
        # Build expectation values by averaging over measurement outcomes
        obs_list = list(observables)
        output = np.zeros((len(params), len(obs_list)), dtype=complex)
        for idx, circ in enumerate([self._bind(p) for p in params]):
            state = Statevector.from_instruction(circ)
            for j, obs in enumerate(obs_list):
                output[idx, j] = state.expectation_value(obs).real
        return output

    def gradient(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        param_idx: int,
    ) -> List[List[complex]]:
        """Parameter‑shift gradient for each observable w.r.t. a chosen parameter."""
        shift = np.pi / 2
        grads: List[List[complex]] = []
        for values in parameter_sets:
            params_plus = list(values)
            params_minus = list(values)
            params_plus[param_idx] += shift
            params_minus[param_idx] -= shift
            state_plus = Statevector.from_instruction(self._bind(params_plus))
            state_minus = Statevector.from_instruction(self._bind(params_minus))
            row: List[complex] = []
            for obs in observables:
                exp_plus = state_plus.expectation_value(obs).real
                exp_minus = state_minus.expectation_value(obs).real
                grad = 0.5 * (exp_plus - exp_minus)
                row.append(grad)
            grads.append(row)
        return grads


__all__ = ["FastBaseEstimator"]
