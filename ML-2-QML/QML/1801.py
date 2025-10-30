"""Advanced FastBaseEstimator for PennyLane with gradient support."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.measurements import MeasurementProcess
from typing import Callable, Iterable, List, Sequence, Tuple, Union


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit with optional gradients."""

    def __init__(self, circuit: qml.QNode, device: str | None = None) -> None:
        """
        Args:
            circuit: A PennyLane QNode that accepts parameters and an `obs` keyword argument.
            device: Optional device string to override the circuit's device (e.g., 'default.qubit.autograd').
        """
        self.circuit = circuit
        self.dev = circuit.device
        self.params = circuit.parameters
        if device is not None:
            # Re‑wrap the circuit on a new device if requested
            self.dev = qml.device(device, wires=circuit.wires)
            self.circuit = qml.qnode(self.dev)(circuit.func)

    def evaluate(
        self,
        observables: Iterable[MeasurementProcess],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[Union[float, complex]]]:
        """
        Compute expectation values for every parameter set and observable.
        Returns a list of lists: outer list over samples, inner over observables.
        """
        observables = list(observables)
        results: List[List[Union[float, complex]]] = []
        for params in parameter_sets:
            row = [self.circuit(*params, obs=obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_batch(
        self,
        observables: Iterable[MeasurementProcess],
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """
        Vectorized evaluation that returns a NumPy array of shape (batch, observables).
        """
        results = self.evaluate(observables, parameter_sets)
        return np.asarray(results, dtype=np.complex128)

    def evaluate_with_gradients(
        self,
        observables: Iterable[MeasurementProcess],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both expectation values and gradients w.r.t. all parameters.

        Returns:
            results: shape (n_samples, n_observables)
            grads:   shape (n_samples, n_params, n_observables)
        """
        observables = list(observables)
        n_samples = len(parameter_sets)
        n_observables = len(observables)

        # Prepare gradient function
        grad_fn = qml.grad(self.circuit, argnum=tuple(range(len(self.params))))

        results = np.empty((n_samples, n_observables), dtype=np.complex128)
        grads = np.empty((n_samples, len(self.params), n_observables), dtype=np.float64)

        for i, params in enumerate(parameter_sets):
            for j, obs in enumerate(observables):
                # Forward value
                val = self.circuit(*params, obs=obs)
                results[i, j] = val

                # Gradient w.r.t. all parameters
                grad = grad_fn(*params, obs=obs)
                grads[i, :, j] = np.asarray(grad, dtype=np.float64)

        return results, grads


class FastEstimator(FastBaseEstimator):
    """
    Adds optional shot‑noise modelling by sampling from the device's measurement distribution.
    The default device must support sampling (e.g., 'default.qubit').
    """

    def evaluate(
        self,
        observables: Iterable[MeasurementProcess],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[Union[float, complex]]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw

        rng = np.random.default_rng(seed)
        noisy: List[List[Union[float, complex]]] = []

        for row in raw:
            noisy_row = [rng.normal(val.real, 1 / shots) + 1j * rng.normal(val.imag, 1 / shots) for val in row]
            noisy.append(noisy_row)

        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
