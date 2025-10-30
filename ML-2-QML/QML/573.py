"""Enhanced quantum estimator using Pennylane.

Features:
* Batched evaluation of expectation values for a list of observables.
* Optional shot‑noise simulation.
* Gradient computation via Pennylane's `grad`.
* Supports any circuit function that builds a circuit given parameters.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple, Callable, Optional
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane import qnode
from pennylane.measurements import Expectation
from pennylane import device as _device

class FastBaseEstimator:
    """Evaluate parametric quantum circuits with expectation values and gradients."""

    def __init__(
        self,
        circuit_builder: Callable[[Sequence[float]], qml.qnode],
        dev: qml.Device | None = None,
    ) -> None:
        """
        Parameters
        ----------
        circuit_builder
            A callable that receives a sequence of parameters and returns a
            Pennylane QNode that returns a statevector.
        dev
            Pennylane device.  Defaults to ``default.qubit`` with 2 wires.
        """
        self._circuit_builder = circuit_builder
        self.dev = dev or _device("default.qubit", wires=2)

    def _build_state_qnode(self, params: Sequence[float]) -> qml.QNode:
        """Return a QNode that outputs the statevector for the given parameters."""
        circuit = self._circuit_builder(params)

        @qml.qnode(self.dev)
        def state_qnode():
            circuit()
            return qml.state()

        return state_qnode

    def _build_qnode_for_observable(
        self, obs: Expectation, params: Sequence[float]
    ) -> qml.QNode:
        """Return a QNode that outputs the expectation value of `obs`."""
        circuit = self._circuit_builder(params)

        @qml.qnode(self.dev)
        def observable_qnode():
            circuit()
            return qml.expval(obs)

        return observable_qnode

    def evaluate(
        self,
        observables: Iterable[Expectation] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        if parameter_sets is None:
            raise ValueError("parameter_sets must be provided")
        observables = list(observables or [qml.expval(qml.PauliZ(0))])
        results: List[List[complex]] = []
        for params in parameter_sets:
            state_qnode = self._build_state_qnode(params)
            state = state_qnode()
            row = [qml.expval(obs, state=state) for obs in observables]
            results.append(row)
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [
                    complex(
                        rng.normal(x.real, max(1e-6, 1 / shots)),
                        rng.normal(x.imag, max(1e-6, 1 / shots)),
                    )
                    for x in row
                ]
                noisy.append(noisy_row)
            results = noisy
        return results

    def evaluate_with_gradients(
        self,
        observables: Iterable[Expectation] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> Tuple[List[List[complex]], List[List[np.ndarray]]]:
        if parameter_sets is None:
            raise ValueError("parameter_sets must be provided")
        observables = list(observables or [qml.expval(qml.PauliZ(0))])
        results: List[List[complex]] = []
        grads: List[List[np.ndarray]] = []
        for params in parameter_sets:
            state_qnode = self._build_state_qnode(params)
            state = state_qnode()
            for obs in observables:
                # Expectation value
                exp_val = qml.expval(obs, state=state)
                results.append([exp_val])
                # Gradient via parameter‑shift rule
                observable_qnode = self._build_qnode_for_observable(obs, params)
                grad_fn = qml.grad(observable_qnode)
                grad = grad_fn(*params)
                grads.append([np.array(grad)])
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy_results = []
            for row in results:
                noisy_row = [
                    complex(
                        rng.normal(x.real, max(1e-6, 1 / shots)),
                        rng.normal(x.imag, max(1e-6, 1 / shots)),
                    )
                    for x in row
                ]
                noisy_results.append(noisy_row)
            results = noisy_results
        return results, grads


__all__ = ["FastBaseEstimator"]
