"""Enhanced FastBaseEstimator for Pennylane QNodes with batched evaluation and gradients.

Features:
* Device selection (default 'default.qubit').
* Batched parameter evaluation.
* Shot noise via Gaussian approximation.
* Parameter‑shift gradient support.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable, List, Sequence

import numpy as np
import pennylane as qml
from pennylane.operation import Operator


class FastBaseEstimator:
    """Evaluate a Pennylane QNode for batched parameters and observables."""

    def __init__(self, qnode: qml.QNode, device: qml.Device | None = None) -> None:
        self.qnode = qnode
        self.device = device or qml.device("default.qubit", wires=qnode.num_wires)
        self.qnode.device = self.device

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables:
            Pennylane Operator instances.
        parameter_sets:
            Iterable of parameter sequences.
        shots:
            If provided, Gaussian noise with std = 1/√shots is added to each mean.
        seed:
            Random seed for reproducibility of shot noise.
        """
        rng = np.random.default_rng(seed)
        results: List[List[complex]] = []

        for params in parameter_sets:
            # Obtain the statevector exactly
            state = qml.state(self.qnode)(*params)
            row = [np.vdot(state, op.matrix() @ state).real for op in observables]
            if shots is not None:
                std = max(1e-6, 1 / np.sqrt(shots))
                row = [rng.normal(val, std) for val in row]
            results.append(row)
        return results

    def grad(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[np.ndarray]]:
        """Return gradients of observables w.r.t. parameters using param‑shift.

        Each returned gradient is a numpy array of shape (len(parameters),).
        """
        grads: List[List[np.ndarray]] = []

        for params in parameter_sets:
            grads_row = []
            for obs in observables:
                grad_fn = qml.gradients.param_shift(self.qnode, argnum=0, observable=obs)
                grad_val = grad_fn(*params)
                grads_row.append(np.array(grad_val))
            grads.append(grads_row)
        return grads


__all__ = ["FastBaseEstimator"]
