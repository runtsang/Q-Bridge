"""Quantum estimator using Pennylane QNodes.

The class now:
* Accepts a Pennylane QNode or a function that returns one.
* Evaluates expectation values for multiple observables.
* Supports shot‑noise simulation and device selection.
* Computes gradients via parameter‑shift or analytic back‑prop.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Tuple

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from pennylane.ops import Operator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""

    def __init__(
        self,
        qnode: Callable[..., qml.QNode] | qml.QNode,
        device: Optional[qml.Device] = None,
        shots: int | None = None,
    ) -> None:
        if isinstance(qnode, qml.QNode):
            self._qnode = qnode
        else:
            self._qnode = qnode()
        self.device = device or qml.device("default.qubit", wires=self._qnode.num_wires)
        self.shots = shots

    def _bind(self, parameter_values: Sequence[float]) -> qml.QNode:
        """Return a new QNode with parameters bound."""
        if len(parameter_values)!= self._qnode.num_params:
            raise ValueError("Parameter count mismatch for bound circuit.")
        return qml.QNode(
            self._qnode.func,
            self.device,
            interface="autograd",
            shots=self.shots,
        )

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[float]] = []
        for values in parameter_sets:
            qnode = self._bind(values)
            state = qnode(*values)
            row = [float(state.expectation_value(obs)) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Add shot‑noise to deterministic evaluations."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def gradient(
        self,
        observable: Operator,
        parameter_set: Sequence[float],
        method: str = "parameter-shift",
    ) -> np.ndarray:
        """Return the gradient of a scalar observable w.r.t. circuit parameters."""
        if method == "parameter-shift":
            return np.array(qml.grad(self._qnode)(*parameter_set))
        elif method == "analytic":
            return np.array(qml.grad(self._qnode, analytic=True)(*parameter_set))
        else:
            raise ValueError(f"Unsupported gradient method: {method}")


__all__ = ["FastBaseEstimator"]
