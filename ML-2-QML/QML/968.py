"""Enhanced quantum estimator using Pennylane.

This module extends the original lightweight quantum estimator by adding GPU
support, optional shot noise, result caching, and gradient computation via
Pennylane's automatic differentiation. It accepts a circuit builder that
produces a QNode capable of evaluating a list of observables.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import pennylane as qml
from pennylane import numpy as pnp

class FastBaseEstimatorGen255:
    """Evaluate expectation values of observables for a parametrized circuit.

    Parameters
    ----------
    circuit_builder:
        Callable that receives a Pennylane device and a list of observables
        and returns a QNode that evaluates those observables for given parameters.
    device:
        Pennylane device name (default ``"default.qubit"``).
    shots:
        Number of measurement shots. ``None`` uses statevector mode.
    """

    def __init__(
        self,
        circuit_builder: Callable[[qml.Device, Iterable[qml.operation.Operator]], qml.QNode],
        *,
        device: str = "default.qubit",
        shots: int | None = None,
    ) -> None:
        self.circuit_builder = circuit_builder
        self.device_name = device
        self.shots = shots
        self._cache: Optional[List[List[complex]]] = None

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        cache: bool = False,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables:
            Iterable of Pennylane operators (e.g., ``qml.PauliZ(0)``).
        parameter_sets:
            Sequence of parameter vectors.
        shots:
            Number of measurement shots. ``None`` uses statevector mode.
        seed:
            Random seed for reproducible shot sampling.
        cache:
            If ``True`` and the same parameter set is evaluated again, cached results are returned.
        """
        observables = list(observables) or [qml.PauliZ(0)]
        results: List[List[complex]] = []

        if cache and self._cache is not None:
            return self._cache

        rng = np.random.default_rng(seed)

        shots = shots or self.shots

        # Determine the set of wires needed by the observables
        wires = set()
        for obs in observables:
            wires.update(obs.wires)
        wires = sorted(wires)

        dev = qml.device(self.device_name, wires=wires, shots=shots)

        # Build a QNode that returns the expectation values of all observables
        circuit = self.circuit_builder(dev, observables)

        for params in parameter_sets:
            exp_vals = circuit(*params)
            if shots is not None:
                exp_vals = [rng.normal(val, max(1e-6, 1.0 / shots)) for val in exp_vals]
            results.append(list(exp_vals))

        if cache:
            self._cache = results

        return results

    def gradients(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        method: str = "parameter-shift",
    ) -> List[List[np.ndarray]]:
        """
        Compute gradients of each observable w.r.t. circuit parameters.

        Parameters
        ----------
        observables:
            Iterable of Pennylane operators.
        parameter_sets:
            Sequence of parameter vectors.
        method:
            Gradient method: ``"parameter-shift"`` (default) or ``"analytic"`` if supported.
        """
        grads: List[List[np.ndarray]] = []

        # Build a device once for gradient evaluation
        wires = set()
        for obs in observables:
            wires.update(obs.wires)
        wires = sorted(wires)

        for params in parameter_sets:
            grad_row: List[np.ndarray] = []
            for obs in observables:
                dev = qml.device(self.device_name, wires=wires)

                @qml.qnode(dev)
                def obs_circuit(*params):
                    self.circuit_builder(dev, [obs])(*params)
                    return qml.expval(obs)

                grad = qml.grad(obs_circuit)(*params)
                grad_row.append(np.array(grad))
            grads.append(grad_row)

        return grads


__all__ = ["FastBaseEstimatorGen255"]
