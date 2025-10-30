"""Enhanced estimator that supports Pennylane QNodes, batching, shot noise, and gradient computation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Union

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane import qnode
from pennylane import device
from pennylane import ops

ScalarObservable = Callable[[pnp.ndarray], pnp.ndarray | float]


class AdvancedHybridEstimator:
    """Estimator for Pennylane QNodes with optional shot noise and analytic gradients.

    Parameters
    ----------
    qnode : qml.QNode
        A Pennylane QNode that accepts a sequence of parameters and returns a state
        or measurement result.  The QNode must be compiled for the chosen device.
    """

    def __init__(self, qnode: qml.QNode) -> None:
        self.qnode = qnode
        self.device = qnode.device
        self.num_params = len(qnode.parameters)

    def evaluate(
        self,
        observables: Iterable[Union[ops.Operator, Callable[[pnp.ndarray], pnp.ndarray]]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Evaluate expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : Iterable[Union[ops.Operator, Callable]]
            Pennylane operators or callables that transform the raw statevector.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors to feed into the QNode.
        shots : int | None, optional
            If provided, sample ``shots`` times and compute the mean to emulate
            measurement shot noise.
        seed : int | None, optional
            Random seed for sampling.

        Returns
        -------
        List[List[complex]]
            Nested list of expectation values for each parameter set.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        rng = np.random.default_rng(seed)

        for params in parameter_sets:
            if len(params)!= self.num_params:
                raise ValueError("Parameter count mismatch for bound circuit.")

            if shots is None:
                # Deterministic expectation via qnode
                state = self.qnode(*params)
                row = [state.expectation_value(obs) if isinstance(obs, ops.Operator)
                       else obs(state) for obs in observables]
            else:
                # Sample-based estimation
                samples = self.qnode(*params, shots=shots, seed=rng.integers(0, 1 << 30))
                # ``samples`` is a tuple of measurement results; convert to statevector
                state = pnp.array(samples)
                row = [state.mean() if isinstance(obs, ops.Operator)
                       else obs(state) for obs in observables]
            results.append(row)

        return results

    def evaluate_gradients(
        self,
        observables: Iterable[Union[ops.Operator, Callable[[pnp.ndarray], pnp.ndarray]]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[np.ndarray]]:
        """Compute gradients of each observable w.r.t. the QNode parameters.

        Parameters
        ----------
        observables : Iterable[Union[ops.Operator, Callable]]
            Pennylane operators or callables that transform the raw statevector.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors to feed into the QNode.

        Returns
        -------
        List[List[np.ndarray]]
            Nested list of gradient arrays (one per observable) for each parameter set.
        """
        observables = list(observables)
        grads: List[List[np.ndarray]] = []

        for params in parameter_sets:
            if len(params)!= self.num_params:
                raise ValueError("Parameter count mismatch for bound circuit.")

            # Use Pennylane's param-shift gradient
            grad_matrix = qml.gradients.param_shift(self.qnode)(*params)
            # ``grad_matrix`` shape: (num_observables, num_params)
            # We need to evaluate each observable separately
            row_grads: List[np.ndarray] = []
            for idx, obs in enumerate(observables):
                # Compute gradient of the specific observable
                grad = grad_matrix[idx] if isinstance(obs, ops.Operator) else grad_matrix[idx]
                row_grads.append(np.array(grad))
            grads.append(row_grads)

        return grads


__all__ = ["AdvancedHybridEstimator"]
