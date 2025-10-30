"""Enhanced estimator for Qiskit circuits with shot‑noise simulation and parameter‑shift gradients.

The class extends the original lightweight QML estimator by adding:
* optional Gaussian shot noise to the exact expectation values
* a parameter‑shift rule for gradient computation
* support for any BaseOperator observables
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit with optional shot noise and gradients."""

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
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of qiskit operators for which expectation values are requested.
        parameter_sets
            Sequence of parameter vectors to evaluate.
        shots
            If provided, add Gaussian noise with variance 1/shots to mimic measurement shot noise.
        seed
            Random seed for reproducibility of shot noise.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [complex(rng.normal(float(val.real), max(1e-6, 1 / shots))) for val in row]
            noisy.append(noisy_row)
        return noisy

    def evaluate_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shift: float = np.pi / 2,
    ) -> List[List[List[complex]]]:
        """Return gradients of observables w.r.t. circuit parameters using the parameter‑shift rule.

        The output shape is [num_parameter_sets][num_observables][num_parameters].
        """
        observables = list(observables)
        gradients: List[List[List[complex]]] = []

        for values in parameter_sets:
            grad_per_set: List[List[complex]] = []
            for observable in observables:
                grad_per_obs: List[complex] = []
                for idx, _ in enumerate(values):
                    shift_plus = list(values)
                    shift_minus = list(values)
                    shift_plus[idx] += shift
                    shift_minus[idx] -= shift
                    exp_plus = self.evaluate([observable], [shift_plus], shots=None)[0][0]
                    exp_minus = self.evaluate([observable], [shift_minus], shots=None)[0][0]
                    grad = 0.5 * (exp_plus - exp_minus) / np.sin(shift)
                    grad_per_obs.append(grad)
                grad_per_set.append(grad_per_obs)
            gradients.append(grad_per_set)
        return gradients


__all__ = ["FastBaseEstimator"]
