"""Quantum‑enhanced estimator utilities using Pennylane.

Features
--------
* Supports batched evaluation of parameter‑shift gradients for expectation values.
* Implements a `FastEstimator` variant that injects shot‑noise into expectation estimates.
* Maintains compatibility with the original API.

The code is intentionally minimal; for heavy‑weight use, consider switching to a simulator backend such as Aer or a real device.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

import pennylane as qml
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ────────────────────────────────────────────────────────────────────────────────

def _ensure_batch(values: Sequence[float]) -> np.ndarray:
    """Convert a list of parameter lists into a 2‑D NumPy array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


# ────────────────────────────────────────────────────────────────────────────────
# Core estimator
# ────────────────────────────────────────────────────────────────────────────────

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""

    def __init__(self, circuit: qml.QNode, device: str | qml.Device | None = None) -> None:
        """
        Parameters
        ----------
        circuit : pennylane.QNode
            A parameter‑dependent quantum circuit.
        device : optional
            Pennylane device to run the circuit on.
            If ``None`` a default 'default.qubit' simulator is used.
        """
        self._device = device or qml.device("default.qubit", wires=circuit.wires)
        self._circuit = qml.QNode(circuit, self._device)
        self._n_params = len(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> qml.QNode:
        if len(parameter_values)!= self._n_params:
            raise ValueError("Parameter count mismatch for bound circuit.")
        # The QNode handles parameter binding internally; we just call it.
        return self._circuit

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator | qml.measurements.MeasurementProcess],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.

        Returns
        -------
        List[List[complex]]
            Shape ``(n_samples, n_observables)``.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            expectation = [
                self._circuit(*values, observable=obs) for obs in observables
            ]
            results.append(expectation)
        return results

    def evaluate_gradients(
        self,
        observables: Iterable[qml.operation.Operator | qml.measurements.MeasurementProcess],
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """
        Compute gradients of each observable with respect to the parameters.

        Uses the parameter‑shift rule.  Returns an array of shape
        ``(n_samples, n_observables, n_params)``.
        """
        observables = list(observables)
        grads: List[np.ndarray] = []
        for obs_index, obs in enumerate(observables):
            # Build a wrapper that returns the chosen observable
            def _obs_wrapper(*params, observable=obs):
                return self._circuit(*params, observable=observable)

            grad_fn = qml.grad(_obs_wrapper)
            grad_vals = [grad_fn(*params) for params in parameter_sets]
            grads.append(np.array(grad_vals))
        return np.stack(grads, axis=1)

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)


# ────────────────────────────────────────────────────────────────────────────────
# Estimator with shot noise
# ────────────────────────────────────────────────────────────────────────────────

class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to deterministic expectation estimates."""

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator | qml.measurements.MeasurementProcess],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for sample in raw:
            noisy_sample = [
                complex(rng.normal(
                    loc=val.real,
                    scale=val.imag / np.sqrt(shots) if val.imag!= 0 else 1.0 / np.sqrt(shots)
                ))
                for val in sample
            ]
            noisy.append(noisy_sample)
        return noisy

    def evaluate_gradients(
        self,
        observables: Iterable[qml.operation.Operator | qml.measurements.MeasurementProcess],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        raw = super().evaluate_gradients(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noise = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(shots),
            size=raw.shape,
        )
        return raw + noise


__all__ = ["FastBaseEstimator", "FastEstimator"]
