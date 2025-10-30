"""Quantum‑classical hybrid estimator using Pennylane.

Extends the original FastBaseEstimator to provide a stochastic or exact
simulation backend, optional shot noise, and a gradient routine.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
import pennylane as qml
from pennylane.measurements import Expectation

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized quantum circuit."""

    def __init__(
        self,
        circuit: qml.QNode,
        device: Optional[qml.Device] = None,
        shots: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        circuit : qml.QNode
            Parameter‑dependent quantum circuit.
        device : qml.Device, optional
            Pennylane device (default: ``default.qubit``).
        shots : int, optional
            Number of shots for stochastic simulation (default: None → exact).
        """
        self.circuit = circuit
        self.device = device or qml.device("default.qubit", wires=circuit.num_wires)
        self.shots = shots
        self.circuit.device = self.device
        self.circuit.shots = shots

    def _bind_and_evaluate(
        self,
        params: Sequence[float],
        observable: Expectation,
    ) -> complex:
        """Bind parameters and evaluate the expectation value."""
        return self.circuit(*params, observable=observable)

    def evaluate(
        self,
        observables: Iterable[Expectation],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            row = [self._bind_and_evaluate(values, obs) for obs in observables]
            results.append(row)
        return results

    def gradient(
        self,
        observables: Iterable[Expectation],
        parameter_sets: Sequence[Sequence[float]],
        *,
        method: str = "parameter_shift",
    ) -> List[List[float]]:
        """Compute gradients of the expectation values using the chosen method."""
        grads: List[List[float]] = []
        for values in parameter_sets:
            grad_row: List[float] = []
            for obs in observables:
                if method == "parameter_shift":
                    grad = qml.grad(self.circuit, argnum=range(len(values)))(*values, observable=obs)
                else:
                    raise ValueError(f"Unsupported gradient method: {method}")
                grad_row.append(float(np.mean(grad)))
            grads.append(grad_row)
        return grads

__all__ = ["FastBaseEstimator"]
