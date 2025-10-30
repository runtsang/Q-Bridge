"""Enhanced quantum estimator using Pennylane for variational circuits."""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import pennylane as qml

ScalarObservable = Callable[[np.ndarray], np.ndarray | complex]


def _ensure_batch(values: Sequence[float]) -> np.ndarray:
    """Convert a 1D sequence of floats into a 2D batch array."""
    array = np.array(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


class FastBaseEstimator:
    """
    Evaluate expectation values of observables for a parametrized quantum circuit.

    Features
    ----------
    * Pennylane QNode integration with custom devices.
    * Support for shot noise via device shots.
    * Automatic gradient computation using the parameter‑shift rule.
    * Batch‑wise evaluation and chunking for large parameter sets.
    """

    def __init__(
        self,
        qnode: qml.QNode,
        *,
        chunk_size: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        qnode : pennylane.QNode
            The QNode representing the parametrized circuit. It must accept a
            single array of parameters and return a state vector or a scalar.
        chunk_size : int, optional
            Number of parameter sets to evaluate in a single batch.
        """
        self._qnode = qnode
        self.chunk_size = chunk_size

    def _bind_and_run(self, params: np.ndarray) -> np.ndarray:
        """Run the QNode for a single parameter set and return the state vector."""
        return self._qnode(params)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the state vector and returns a complex
            expectation value. If empty, the state vector itself is returned.
        parameter_sets : sequence of sequences
            Each inner sequence contains the float parameters for a single
            circuit execution.

        Returns
        -------
        List[List[complex]]
            Nested list with shape (len(parameter_sets), len(observables)).
        """
        observables = list(observables) or [lambda state: state]
        results: List[List[complex]] = []

        # Handle large batches by chunking
        batch_indices = (
            range(0, len(parameter_sets), self.chunk_size)
            if self.chunk_size
            else [0]
        )

        for start in batch_indices:
            end = start + self.chunk_size if self.chunk_size else len(parameter_sets)
            batch = _ensure_batch(parameter_sets[start:end])
            for params in batch:
                state = self._bind_and_run(params)
                row: List[complex] = []
                for observable in observables:
                    value = observable(state)
                    if isinstance(value, np.ndarray):
                        scalar = np.mean(value).item()
                    else:
                        scalar = value
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate_gradient(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[np.ndarray]]:
        """
        Compute gradients of each observable with respect to the circuit parameters
        using the parameter‑shift rule.

        Returns
        -------
        List[List[np.ndarray]]
            Nested list with shape (len(parameter_sets), len(observables)).
            Each array has the same shape as the parameter set.
        """
        observables = list(observables) or [lambda state: state]
        grads: List[List[np.ndarray]] = []

        for params in parameter_sets:
            grad_funcs = [
                qml.grad(lambda p: observable(self._qnode(p)))(params)
                for observable in observables
            ]
            grads.append(grad_funcs)
        return grads

    def add_shot_noise(
        self,
        results: List[List[complex]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Add Gaussian shot noise to deterministic results.

        Parameters
        ----------
        results : List[List[complex]]
            Deterministic evaluation outputs.
        shots : int, optional
            Number of shots; if None, no noise is added.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        List[List[complex]]
            Noisy results.
        """
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                rng.normal(np.real(mean), max(1e-6, 1 / shots))
                + 1j * rng.normal(np.imag(mean), max(1e-6, 1 / shots))
                for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


class FastEstimator(FastBaseEstimator):
    """
    Adds optional Gaussian shot noise to the deterministic estimator.

    The constructor mirrors that of :class:`FastBaseEstimator`; the noise
    is applied via :meth:`evaluate` by passing ``shots`` and ``seed``.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        return self.add_shot_noise(raw, shots, seed)


__all__ = ["FastBaseEstimator", "FastEstimator"]
