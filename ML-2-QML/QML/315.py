"""Enhanced FastBaseEstimator for PennyLane circuits with shot‑noise, batched observables and gradient support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional, Union

import numpy as np
import pennylane as qml
from pennylane import QuantumDevice
from pennylane import QNode

class FastBaseEstimatorGen350:
    """Quantum estimator that evaluates expectation values for a parameterised PennyLane circuit.

    Parameters
    ----------
    circuit
        A :class:`pennylane.QNode` that maps parameters to a quantum state.
    device
        Optional PennyLane device.  If ``None`` the circuit's device is used.

    Features
    --------
    * Vectorised evaluation across many parameter sets.
    * Supports arbitrary observables (Pauli strings, custom operators).
    * Shot‑noise simulation via the device's measurement noise model.
    * Automatic gradient calculation using the parameter‑shift rule.
    * Backend agnostic: any PennyLane device can be supplied.
    """

    def __init__(self, circuit: QNode, device: Optional[QuantumDevice] = None) -> None:
        self.circuit = circuit
        self.device = device or circuit.device

    def evaluate(
        self,
        observables: Iterable[Union[qml.operation.Operator, qml.measurements.Expectation, qml.measurements.Variance]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
    ) -> List[List[complex]]:
        """
        Evaluate expectation values for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of PennyLane operators or measurement objects.
        parameter_sets
            Iterable of parameter vectors.
        shots
            Number of shots to simulate.  If ``None`` uses the device's default.
        """
        if shots is not None:
            self.device.shots = shots

        results: List[List[complex]] = []
        for params in parameter_sets:
            row: List[complex] = []
            for obs in observables:
                def qnode_wrapper(*ps):
                    self.circuit(*ps)
                    return qml.expval(obs)
                qnode = qml.QNode(qnode_wrapper, self.device)
                val = qnode(*params)
                row.append(val)
            results.append(row)
        return results

    def gradients(
        self,
        observables: Iterable[Union[qml.operation.Operator, qml.measurements.Expectation, qml.measurements.Variance]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[np.ndarray]]]:
        """
        Compute gradients of all observables w.r.t. circuit parameters.

        Returns
        -------
        grads
            Nested list: outer list over parameter sets, inner list over observables,
            inner‑inner list contains NumPy arrays for each circuit parameter.
        """
        grads: List[List[List[np.ndarray]]] = []
        for params in parameter_sets:
            row_grads: List[List[np.ndarray]] = []
            for obs in observables:
                def qnode_wrapper(*ps):
                    self.circuit(*ps)
                    return qml.expval(obs)
                qnode = qml.QNode(qnode_wrapper, self.device)
                grad_fn = qml.grad(qnode)
                grad = grad_fn(*params)
                row_grads.append(grad)
            grads.append(row_grads)
        return grads


__all__ = ["FastBaseEstimatorGen350"]
