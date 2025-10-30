"""Fast quantum estimator based on Pennylane with support for gradients and shot noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import pennylane as qml
import pennylane.math as qml_math
import numpy as np

# Type alias for Pennylane observables
Observable = qml.operation.Operator


class FastBaseEstimator:
    """Evaluate expectation values of a parameterised Pennylane circuit.

    Parameters
    ----------
    circuit_builder:
        Callable that accepts a list of parameters and builds a Pennylane circuit.
        It can return a QNode directly or a function that applies gates and returns
        a state vector via ``qml.state()``.
    wires:
        Number of wires used by the circuit.
    device:
        Pennylane device; defaults to the CPU simulator.
    """

    def __init__(
        self,
        circuit_builder: qml.QNode | callable,
        wires: int,
        device: Optional[qml.Device] = None,
    ) -> None:
        self.wires = wires
        self.device = device or qml.device("default.qubit", wires=wires)

        if isinstance(circuit_builder, qml.QNode):
            self._qnode = circuit_builder
        else:
            # Wrap the user function to produce a state vector
            def _wrapped(params):
                circuit_builder(params)
                return qml.state()

            self._qnode = qml.QNode(_wrapped, self.device)

    def _bind(self, parameter_values: Sequence[float]) -> List[float]:
        """Return parameters in the order expected by the QNode."""
        return list(parameter_values)

    def evaluate(
        self,
        observables: Iterable[Observable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            state = self._qnode(*self._bind(params))
            row = [qml_math.expval(obs, state).item() for obs in observables]
            results.append(row)
        return results

    def gradients(
        self,
        observables: Iterable[Observable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[float]]]:
        """Return gradients of each observable w.r.t. circuit parameters."""
        observables = list(observables)
        grads: List[List[List[float]]] = []

        for params in parameter_sets:
            grad_row: List[List[float]] = []
            for obs in observables:
                grad = qml.gradients.jacobian(
                    lambda *pars: qml_math.expval(obs, self._qnode(*pars)), 0
                )(*self._bind(params))
                grad_row.append(grad.tolist())
            grads.append(grad_row)
        return grads


class FastEstimator(FastBaseEstimator):
    """Estimator that adds optional shot noise to expectation values."""

    def evaluate(
        self,
        observables: Iterable[Observable],
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
            noisy_row = [rng.normal(float(val), max(1e-6, 1 / shots)) for val in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
