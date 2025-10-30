"""Quantum estimator built on PennyLane.

Supports batched evaluation of expectation values for a parameterised
circuit on any PennyLane device.  The class exposes a shot‑limited
interface that can be used to emulate finite‑sample measurements.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import pennylane as qml
import numpy as np

Observable = qml.operation.Operator


class FastBaseEstimator:
    """Evaluate expectation values of a PennyLane circuit.

    Parameters
    ----------
    circuit
        A PennyLane QNode that takes a 1‑D array of parameters and returns a
        state vector or measurement.
    device
        Name of a PennyLane device or an instantiated ``qml.Device``.
        If a name is supplied, a default qubit simulator is created with
        ``shots=None`` (exact expectation values).
    shots
        Number of measurement shots.  ``None`` requests the device's
        default exact evaluation.
    """

    def __init__(
        self,
        circuit: qml.QNode,
        device: str | qml.Device | None = None,
        shots: int | None = None,
    ) -> None:
        if device is None:
            device = qml.device("default.qubit", wires=circuit.wires, shots=shots)
        elif isinstance(device, str):
            device = qml.device(device, wires=circuit.wires, shots=shots)
        self.circuit = circuit
        self.circuit.device = device

    def _bind(self, params: Sequence[float]) -> qml.QNode:
        """Return a new QNode with parameters bound to ``params``."""
        return qml.QNode(lambda: self.circuit(*params), device=self.circuit.device)

    def evaluate(
        self,
        observables: Iterable[Observable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable.

        The method uses ``qml.execute`` to evaluate all observables in a
        single shot (exact or sampled depending on ``device.shots``).
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            # Build a QNode that returns all expectation values
            def circuit_wrapper():
                return [qml.expval(obs) for obs in observables]

            bound_qnode = qml.QNode(circuit_wrapper, device=self.circuit.device)
            expectation = qml.execute(
                [bound_qnode], [params], device=self.circuit.device
            )[0]
            results.append(expectation)

        return results

    def evaluate_shots(
        self,
        observables: Iterable[Observable],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
    ) -> List[List[complex]]:
        """Same as :meth:`evaluate` but forces a finite number of shots."""
        # Temporarily override device shots
        original_shots = self.circuit.device.shots
        self.circuit.device.shots = shots
        try:
            return self.evaluate(observables, parameter_sets)
        finally:
            self.circuit.device.shots = original_shots
