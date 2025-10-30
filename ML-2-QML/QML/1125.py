"""Hybrid quantum estimator built on Pennylane that supports shot‑noise and analytic gradients."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.devices import Device
from pennylane.operation import Observable
from pennylane.measurements import Expectation

ScalarObservable = Callable[[qml.QNode], Observable]


class FastBaseEstimator:
    """
    Evaluate expectation values of a parameterized quantum circuit.

    Parameters
    ----------
    circuit : qml.QNode
        A Pennylane QNode that returns a list of expectation values
        corresponding to the observables passed during construction.
    device : str | Device | None, optional
        Target device name or instance; defaults to ``"default.qubit"``.
    shots : int | None, optional
        Number of shots for the device; ``None`` → exact simulation.
    seed : int | None, optional
        RNG seed for device initialization.

    Notes
    -----
    * The estimator supports batched evaluation via ``evaluate``.
    * ``compute_gradients`` returns analytic gradients using the
      parameter‑shift rule.
    """

    def __init__(
        self,
        circuit: qml.QNode,
        device: str | Device | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        if isinstance(device, Device):
            self._device = device
        else:
            self._device = qml.device(device or "default.qubit", wires=circuit.wires, shots=shots, seed=seed)

        # Wrap the circuit to ensure it runs on the chosen device
        self._circuit = qml.QNode(circuit.func, self._device, interface="numpy")

    def evaluate(
        self,
        observables: Iterable[Observable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        if not observables:
            raise ValueError("At least one observable must be provided.")

        # Build a new QNode that returns the desired observables
        @qml.qnode(self._device, interface="numpy")
        def _wrapped(*params):
            self._circuit(*params)  # execute original circuit
            return [qml.expval(obs) for obs in observables]

        results: List[List[complex]] = []
        for params in parameter_sets:
            row = _wrapped(*params)
            results.append(row)

        return np.array(results, dtype=np.complex128)

    def compute_gradients(
        self,
        observables: Iterable[Observable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[np.ndarray]:
        """
        Compute analytic gradients of the summed observable loss w.r.t. circuit parameters.

        Returns a list of gradient arrays, one per parameter set.  Each gradient
        array has shape ``(n_params, n_observables)``.
        """
        observables = list(observables)
        if not observables:
            raise ValueError("At least one observable must be provided.")

        @qml.qnode(self._device, interface="numpy")
        def _wrapped(*params):
            self._circuit(*params)
            return [qml.expval(obs) for obs in observables]

        grads: List[np.ndarray] = []
        for params in parameter_sets:
            grad = qml.gradients.param_shift(_wrapped, argnum=0)(*params)
            grads.append(np.array(grad, dtype=np.complex128))

        return grads

    def evaluate_with_shots(
        self,
        observables: Iterable[Observable],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """Same as ``evaluate`` but forces a shot‑based simulation."""
        device = qml.device(self._device.name, wires=self._device.wires, shots=shots, seed=seed)
        @qml.qnode(device, interface="numpy")
        def _wrapped(*params):
            self._circuit(*params)
            return [qml.expval(obs) for obs in observables]

        results: List[List[complex]] = []
        for params in parameter_sets:
            results.append(_wrapped(*params))

        return np.array(results, dtype=np.complex128)


__all__ = ["FastBaseEstimator"]
