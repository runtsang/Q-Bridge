"""Quantum FastBaseEstimator with gradient and shot‑noise support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

SHIFT = np.pi / 2  # Parameter‑shift step for rotation gates


class FastBaseEstimator:
    """Evaluate a parametrized quantum circuit and its derivatives."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = Aer.get_backend("statevector_simulator")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute exact expectation values for each parameter set and observable.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound_circ = self._bind(values)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        return results

    def evaluate_expectations(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Alias for :meth:`evaluate`."""
        return self.evaluate(observables, parameter_sets)

    def evaluate_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[np.ndarray]]:
        """
        Estimate gradients of each observable w.r.t. circuit parameters
        using the parameter‑shift rule.

        Returns a list of lists of NumPy arrays; the outer list corresponds to
        parameter sets, the inner list to observables, and each array is the
        gradient vector.
        """
        observables = list(observables)
        grads: List[List[np.ndarray]] = []

        for values in parameter_sets:
            grad_obs: List[np.ndarray] = []

            for observable in observables:
                grad: List[complex] = []

                for idx in range(len(self._parameters)):
                    forward = list(values)
                    backward = list(values)
                    forward[idx] += SHIFT
                    backward[idx] -= SHIFT

                    val_plus = self.evaluate_expectations(
                        [observable], [forward]
                    )[0][0]
                    val_minus = self.evaluate_expectations(
                        [observable], [backward]
                    )[0][0]

                    grad.append((val_plus - val_minus) / (2 * np.sin(SHIFT)))

                grad_obs.append(np.array(grad, dtype=complex))

            grads.append(grad_obs)

        return grads

    def evaluate_shot_noise(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """
        Simulate finite‑shot measurements by adding Gaussian noise to the
        exact expectation values.  The noise variance is 1/shots, matching
        the quantum projection noise of a projective measurement.
        """
        rng = np.random.default_rng(seed)
        exact = self.evaluate(observables, parameter_sets)
        noisy = []
        for row in exact:
            noisy_row = [
                float(rng.normal(val, max(1e-6, 1 / shots))) for val in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator"]
