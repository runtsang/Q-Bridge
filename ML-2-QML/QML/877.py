"""Enhanced FastBaseEstimator for PennyLane circuits with shot noise and gradient support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import pennylane as qml
import numpy as np

class FastBaseEstimator:
    """Evaluate PennyLane quantum circuits for parameter sets with optional shot noise and gradients."""

    def __init__(self, circuit: qml.QNode, device: qml.Device | None = None) -> None:
        self.circuit = circuit
        self.device = device or circuit.device

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        return_gradients: bool = False,
    ) -> List[List[complex]]:
        """Compute expectation values (and optionally gradients) for each parameter set.

        Args:
            observables: Iterable of PennyLane operators (e.g., qml.Hermitian, qml.PauliZ).
            parameter_sets: Iterable of parameter vectors.
            shots: If provided, samples expectation values with finite shots.
            seed: RNG seed for reproducibility.
            return_gradients: If True, returns gradients of expectation values w.r.t. parameters.

        Returns:
            A list of rows, each containing expectation values (and optionally gradients) for a parameter set.
        """
        observables = list(observables) or [qml.Identity]
        rng = np.random.default_rng(seed)
        results: List[List[complex]] = []

        for params in parameter_sets:
            if shots is None:
                expectations = self.circuit(*params, observables=observables)
            else:
                expectations = self.circuit(*params, observables=observables, shots=shots)

            row: List[complex] = []
            for exp in expectations:
                if shots is not None:
                    noise = rng.normal(0, max(1e-6, 1 / np.sqrt(shots)))
                    exp = exp + noise
                row.append(exp)

            if return_gradients:
                grad_function = qml.grad(self.circuit, argnum=0)
                grads = grad_function(*params, observables=observables)
                row.extend(grads)

            results.append(row)

        return results


__all__ = ["FastBaseEstimator"]
