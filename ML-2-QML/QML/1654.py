"""Hybrid estimator for parameterised quantum circuits with noise and gradient support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import pennylane as qml
from pennylane import operation


class FastHybridEstimator:
    """Evaluate a Pennylane QNode for a set of parameters and observables.

    Features
    --------
    * Shot‑noise simulation with configurable shot count.
    * Gradient computation via Pennylane's parameter‑shift rule.
    * Supports complex‑valued expectation values.
    """

    def __init__(self, circuit: qml.QNode) -> None:
        self.circuit = circuit
        self.n_params = circuit.num_params

    def _check_params(self, params: Sequence[float]) -> np.ndarray:
        if len(params)!= self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")
        return np.array(params, dtype=np.float64)

    def evaluate(
        self,
        observables: Iterable[operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Parameters
        ----------
        observables
            Iterable of Pennylane observables.
        parameter_sets
            Sequence of parameter vectors for the circuit.
        shots
            If provided, adds Gaussian noise with std=1/sqrt(shots).
        seed
            Random seed for reproducible noise.

        Returns
        -------
        List of [batch][observable] expectation values.
        """
        results: List[List[complex]] = []

        for params in parameter_sets:
            params = self._check_params(params)
            state = self.circuit(*params)
            row = [qml.expval(obs, state) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy = []
        for row in results:
            noisy.append(
                [
                    rng.normal(val.real, max(1e-6, 1 / shots))
                    + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                    for val in row
                ]
            )
        return noisy

    def evaluate_with_gradients(
        self,
        observables: Iterable[operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[List[List[complex]], List[List[List[complex]]]]:
        """
        Compute expectation values and gradients for each observable.

        Returns
        -------
        results
            List of [batch][observable] expectation values.
        gradients
            List of [batch][observable][param] gradients (complex).
        """
        results: List[List[complex]] = []
        gradients: List[List[List[complex]]] = []

        for params in parameter_sets:
            params = self._check_params(params)
            row: List[complex] = []
            grad_row: List[List[complex]] = []

            for obs in observables:
                func = lambda *p, obs=obs: qml.expval(obs, self.circuit(*p))
                val = func(*params)
                grad = qml.grad(func, argnum=list(range(self.n_params)))(*params)
                row.append(val)
                grad_row.append(grad)

            results.append(row)
            gradients.append(grad_row)

        return results, gradients


__all__ = ["FastHybridEstimator"]
