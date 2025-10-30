"""FastEstimator: extend quantum estimator with variational circuits and gradient support.

The class accepts a PennyLane circuit function that returns a statevector and
evaluates expectation values of observables across batches of parameter
vectors.  It now also provides analytic gradients via PennyLane’s automatic
differentiation and supports shot‑noise simulation by configuring the device’s
``shots`` attribute.  The API mirrors the ML variant for consistency.

Typical usage:

>>> import pennylane as qml
>>> dev = qml.device('default.qubit', wires=1, shots=1024)
>>> @qml.qnode(dev)
... def circuit(params):
...     qml.RX(params[0], wires=0)
...     return qml.state()
>>> estimator = FastEstimator(circuit, dev)
>>> results = estimator.evaluate([qml.PauliZ(0)], [[0.5], [1.0]])
>>> grads = estimator.evaluate_gradients([qml.PauliZ(0)], [[0.5], [1.0]])
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import pennylane as qml
import pennylane.numpy as np
from pennylane import numpy as npy
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.devices import Device

Observable = Operator | MeasurementProcess


class FastEstimator:
    """Evaluate a PennyLane circuit for batches of parameters and observables.

    Parameters
    ----------
    circuit_func : Callable
        A PennyLane circuit function that accepts a sequence of floats and
        returns a statevector (via ``qml.state()``).  The function must be
        decorated with ``@qml.qnode`` or wrapped later.
    device : Device, optional
        The PennyLane device used by the circuit.  If omitted, a default
        ``default.qubit`` device with one wire is created.
    """

    def __init__(self, circuit_func: Callable[..., np.ndarray], device: Device | None = None) -> None:
        if device is None:
            device = qml.device("default.qubit", wires=1)
        self.circuit_func = circuit_func
        self.device = device
        self.qnode = qml.QNode(circuit_func, self.device)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    def _expectation(self, state: np.ndarray, observable: Observable) -> complex:
        """Compute the expectation value of an observable on a given statevector."""
        mat = qml.matrix(observable, device=self.device)
        return np.vdot(state, mat @ state)

    def evaluate(
        self,
        observables: Iterable[Observable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return expectation values for each observable and parameter set.

        Parameters
        ----------
        observables
            Iterable of PennyLane operators or measurement processes.  If empty,
            the circuit must return a single expectation value.
        parameter_sets
            Sequence of parameter vectors; each vector is a list/tuple of floats.
        """
        observables = list(observables) or [self.qnode]
        results: List[List[complex]] = []

        for params in parameter_sets:
            state = self.qnode(*params)
            row = [self._expectation(state, obs) for obs in observables]
            results.append(row)
        return results

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------
    def _grad_qnode(self, observable: Observable) -> Callable[..., np.ndarray]:
        """Return a QNode that outputs the expectation of ``observable``."""
        @qml.qnode(self.device)
        def grad_qnode(*params):
            state = self.circuit_func(*params)
            mat = qml.matrix(observable, device=self.device)
            return np.vdot(state, mat @ state)
        return grad_qnode

    def evaluate_gradients(
        self,
        observables: Iterable[Observable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[np.ndarray]]:
        """Return analytic gradients of each observable w.r.t the parameters.

        The gradients are returned as NumPy arrays of shape ``(len(params),)``.
        """
        observables = list(observables) or [self.qnode]
        grad_results: List[List[np.ndarray]] = []

        for params in parameter_sets:
            row_grads: List[np.ndarray] = []
            for obs in observables:
                grad_qnode = self._grad_qnode(obs)
                grad_func = qml.gradients.param_shift(grad_qnode)
                grad = grad_func(*params)  # Tuple of gradients per parameter
                grad_array = np.array(grad).reshape(-1)
                row_grads.append(grad_array.copy())
            grad_results.append(row_grads)
        return grad_results

    # ------------------------------------------------------------------
    # Shot‑noise simulation
    # ------------------------------------------------------------------
    def evaluate_with_shots(
        self,
        observables: Iterable[Observable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Wrap :meth:`evaluate` with finite‑shot noise.

        Parameters
        ----------
        shots
            Number of measurement shots.  If ``None`` the deterministic device
            configuration is used.
        seed
            Random seed for reproducibility of the stochastic device.
        """
        if shots is not None:
            self.device.shots = shots
            if seed is not None:
                np.random.seed(seed)
        return self.evaluate(observables, parameter_sets)


__all__ = ["FastEstimator"]
