"""Enhanced quantum estimator using Pennylane with shot noise, gradients, and training."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane.operation import Observable
from collections.abc import Sequence, Iterable
from typing import List, Callable, Tuple, Optional

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized Pennylane circuit."""

    def __init__(
        self,
        circuit: Callable[..., None],
        wires: Sequence[int],
        shots: int | None = None,
        device_name: str = "default.qubit",
    ) -> None:
        """
        Parameters
        ----------
        circuit:
            A Pennylane function that accepts parameters and an optional observable.
        wires:
            Sequence of wire indices used by the circuit.
        shots:
            Number of shots for a probabilistic simulation; if None use exact statevector.
        device_name:
            Pennylane device; defaults to "default.qubit".
        """
        self.wires = wires
        self.shots = shots
        self.device = qml.device(device_name, wires=wires, shots=shots)
        self.circuit = qml.QNode(circuit, self.device)

    def evaluate(
        self,
        observables: Iterable[Observable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return a list of expectation values for each parameter set and observable."""
        results: List[List[complex]] = []
        for params in parameter_sets:
            row = [self.circuit(*params, observable=o) for o in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds shot noise simulation, gradient via parameter-shift, and simple training."""

    def evaluate(
        self,
        observables: Iterable[Observable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[List[complex]]:
        """Evaluate with optional shot noise.  If `shots` is None, use the
        estimator's device configuration."""
        if shots is None:
            shots = self.shots
        if shots is None:
            return super().evaluate(observables, parameter_sets)
        rng = np.random.default_rng()
        results: List[List[complex]] = []
        for params in parameter_sets:
            row = []
            for o in observables:
                mean = float(self.circuit(*params, observable=o))
                std = np.sqrt(mean * (1 - mean)) / np.sqrt(shots) if 0 <= mean <= 1 else 0
                noisy = rng.normal(mean, std)
                row.append(noisy)
            results.append(row)
        return results

    def gradient(
        self,
        observables: Iterable[Observable],
        parameter_set: Sequence[float],
        param_index: int,
    ) -> float:
        """Compute gradient of the first observable w.r.t a single circuit parameter
        using the parameter-shift rule."""
        observable = list(observables)[0]
        shift = np.pi / 2
        params_plus = list(parameter_set)
        params_minus = list(parameter_set)
        params_plus[param_index] += shift
        params_minus[param_index] -= shift
        forward = float(self.circuit(*params_plus, observable=observable))
        backward = float(self.circuit(*params_minus, observable=observable))
        grad = (forward - backward) / 2
        return grad

    def train(
        self,
        loss_fn: Callable[[List[complex], List[complex]], float],
        optimizer: Callable[[Sequence[float], List[float]], Sequence[float]],
        data_loader: Iterable[Tuple[Sequence[float], List[complex]]],
        epochs: int = 1,
        shots: int | None = None,
    ) -> None:
        """Simple gradient-based training loop for the parametrized circuit.

        Parameters
        ----------
        loss_fn:
            Loss function that takes (predictions, targets) and returns a scalar loss.
        optimizer:
            Function that takes current parameters and gradients and returns updated parameters.
        data_loader:
            Iterable of (parameter_set, target_observables) tuples.
        epochs:
            Number of training epochs.
        shots:
            Number of shots for stochastic evaluation; if None use exact.
        """
        if shots is None:
            shots = self.shots
        for _ in range(epochs):
            for params, targets in data_loader:
                preds = [self.circuit(*params, observable=o) for o in targets]
                loss = loss_fn(preds, targets)
                grads = []
                for i in range(len(params)):
                    grads.append(self.gradient(list(targets), params, i))
                params = optimizer(params, grads)
