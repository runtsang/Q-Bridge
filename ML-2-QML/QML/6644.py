"""Enhanced quantum estimator using Pennylane with shot noise and gradient support."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import pennylane as qml
import numpy as np

ScalarObservable = Callable[[qml.numpy.array], complex]
GradObservable = Callable[[qml.numpy.array], complex]


class FastBaseEstimator:
    """Evaluate a parameterized quantum circuit for multiple parameter sets and observables.

    Supports shot noise simulation and analytic gradients via the parameter‑shift rule.
    The circuit is supplied as a callable that, given a sequence of parameters,
    applies gates to a Pennylane quantum tape.
    """

    def __init__(
        self,
        circuit_builder: Callable[[Sequence[float]], None],
        wires: int = 1,
        device: str = "default.qubit",
        shots: int | None = None,
    ) -> None:
        self._circuit_builder = circuit_builder
        self._device = qml.device(device, wires=wires)
        self.shots = shots

    def _make_qnode(self, observable: qml.operation.Operator) -> qml.QNode:
        """Create a QNode that returns the expectation value of ``observable``."""
        @qml.qnode(self._device, shots=self.shots)
        def qnode(*params):
            self._circuit_builder(params)
            return qml.expval(observable)
        return qnode

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            row: List[complex] = []
            for obs in observables:
                qnode = self._make_qnode(obs)
                val = qnode(*params)
                row.append(val)
            results.append(row)
        return results

    def evaluate_shots(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
    ) -> List[List[complex]]:
        """Same as ``evaluate`` but uses the specified shot count."""
        self.shots = shots
        return self.evaluate(observables, parameter_sets)

    def evaluate_gradients(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[Tuple[np.ndarray,...]]]:
        """Compute analytic gradients of each observable w.r.t. circuit parameters.

        Returns a list of tuples, one per observable, containing the gradient
        arrays for each parameter.  The gradients are returned as NumPy arrays.
        """
        observables = list(observables)
        grads_list: List[List[Tuple[np.ndarray,...]]] = []
        for params in parameter_sets:
            row: List[Tuple[np.ndarray,...]] = []
            for obs in observables:
                qnode = self._make_qnode(obs)
                grad_fn = qml.grad(qnode)
                grads = grad_fn(*params)
                grads_np = tuple(g.numpy() for g in grads)
                row.append(grads_np)
            grads_list.append(row)
        return grads_list


class FastEstimator(FastBaseEstimator):
    """Adds shot‑noise simulation to the deterministic quantum estimator.

    The shot noise is implemented by running the circuit with the specified
    number of shots and returning the mean of the measurement outcomes.
    """

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                rng.normal(val.real, 1 / np.sqrt(shots)) + 1j * rng.normal(val.imag, 1 / np.sqrt(shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
