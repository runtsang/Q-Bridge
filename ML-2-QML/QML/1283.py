"""Hybrid estimator that wraps a Pennylane variational circuit and supports shot noise and gradient estimation.

This class mirrors the interface of the classical HybridEstimator so that a
hybrid workflow can swap the two implementations without changing the API.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence

import pennylane as qml
from pennylane import numpy as pnp


class FastBaseEstimator:
    """Base class from the seed – unchanged."""
    def __init__(self, circuit_fn: Callable[..., qml.QNode]) -> None:
        """
        Parameters
        ----------
        circuit_fn
            A function that, given parameters, builds and returns a QNode that
            outputs a Statevector. The function should be pure and can be
            called repeatedly.
        """
        self._circuit_fn = circuit_fn
        # Create a device once; the number of wires is inferred from the first call
        dummy_params = [0.0] * 5  # placeholder; will be overridden
        qnode = circuit_fn(*dummy_params)
        self._device = qnode.device

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        results: List[List[complex]] = []
        for params in parameter_sets:
            qnode = self._circuit_fn(*params)
            state = qnode()
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class HybridEstimator(FastBaseEstimator):
    """Hybrid estimator that adds shot noise and a simple gradient estimator."""

    def __init__(
        self,
        circuit_fn: Callable[..., qml.QNode],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(circuit_fn)
        self.shots = shots
        self.rng = np.random.default_rng(seed)

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        shots_to_use = shots if shots is not None else self.shots
        if shots_to_use is None:
            return raw
        rng = np.random.default_rng(seed if seed is not None else None)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                rng.normal(row_val.real, 1 / shots_to_use) + 1j * rng.normal(row_val.imag, 1 / shots_to_use)
                for row_val in row
            ]
            noisy.append(noisy_row)
        return noisy

    def gradient(
        self,
        observable: qml.operation.Operator,
        parameter_values: Sequence[float],
    ) -> np.ndarray:
        """Compute the gradient of an expectation value using the parameter‑shift rule."""
        # Build a QNode that returns the expectation of the given observable
        @qml.qnode(self._device)
        def qnode(*params):
            self._circuit_fn(*params)
            return qml.expval(observable)

        grad_fn = qml.grad(qnode)
        return grad_fn(*parameter_values)

__all__ = ["HybridEstimator"]
