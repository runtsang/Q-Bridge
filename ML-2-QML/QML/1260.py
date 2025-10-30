"""Enhanced quantum estimator using Pennylane's hybrid quantum‑classical framework.

Features
--------
- Variational circuit support via Pennylane QNodes.
- Expectation value evaluation for multiple observables.
- Gradient computation using parameter‑shift rule or automatic differentiation.
- Shot noise simulation on top of expectation values.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Iterable, List, Sequence, Tuple, Optional

import pennylane as qml
from pennylane import numpy as pnp

ScalarObservable = Callable[[np.ndarray], np.ndarray | float]
GradientFunction = Callable[[np.ndarray], np.ndarray]


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized Pennylane circuit."""

    def __init__(
        self,
        qnode: qml.QNode,
        device: Optional[qml.Device] = None,
        shots: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        qnode : qml.QNode
            Pennylane QNode representing the variational circuit.
        device : qml.Device, optional
            Quantum device; defaults to a default Aer simulator.
        shots : int, optional
            Number of shots for stochastic simulation; if None, use analytical expectation.
        """
        self.qnode = qnode
        self.device = device or qml.device("default.qubit", wires=qnode.wires)
        self.shots = shots

    def _run(self, params: Sequence[float]) -> np.ndarray:
        """Execute the QNode and return the state vector or sampled expectation."""
        if self.shots is None:
            return self.qnode(*params)
        else:
            return self.qnode(*params, shots=self.shots)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            state = self._run(params)
            row = [obs(state) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_grad(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        grad_fn: GradientFunction | None = None,
    ) -> Tuple[List[List[complex]], List[List[complex]]]:
        """Evaluate observables and compute gradients via parameter‑shift rule.

        Parameters
        ----------
        grad_fn : GradientFunction, optional
            Custom gradient function; if None, use Pennylane's default.
        """
        observables = list(observables)
        obs_results: List[List[complex]] = []
        grad_results: List[List[complex]] = []

        for params in parameter_sets:
            expectation = self._run(params)
            obs_row = [obs(expectation) for obs in observables]
            obs_results.append(obs_row)

            if grad_fn is not None:
                grads = grad_fn(expectation)
            else:
                grads = qml.grad(self.qnode)(*params)
            grad_results.append(grads)

        return obs_results, grad_results


__all__ = ["FastBaseEstimator"]
