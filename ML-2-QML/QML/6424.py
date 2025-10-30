"""Quantum FastBaseEstimator leveraging Pennylane for variational circuit evaluation."""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence

import pennylane as qml
import numpy as np


class FastBaseEstimator:
    """Quantum estimator that evaluates expectation values of a parameterized Pennylane circuit.

    Parameters
    ----------
    circuit:
        A function that builds a Pennylane circuit.  It should not return a value
        but should call Pennylane operations to modify the device.
    device:
        Pennylane device name, e.g. ``'default.qubit'``.
    shots:
        Number of shots for stochastic estimation.  If ``None``, the device returns
        exact expectation values.
    """

    def __init__(self, circuit: Callable[..., None], device: str = "default.qubit", shots: int | None = None) -> None:
        self.circuit = circuit
        self.device = device
        self.shots = shots

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables:
            Iterable of Pennylane operators for which the expectation value is
            computed.
        parameter_sets:
            Sequence of parameter vectors.  Each vector is fed to the circuit as
            a single input example.

        Returns
        -------
        List[List[complex]]
            Nested list of expectation values; outer list corresponds to
            ``parameter_sets`` and inner list corresponds to ``observables``.
        """
        shots = self.shots or 0

        @qml.qnode(self.device, interface="autograd")
        def _multi_obs_node(*params: float) -> np.ndarray:
            self.circuit(*params)
            return [qml.expval(obs) for obs in observables]

        results: List[List[complex]] = []
        for params in parameter_sets:
            flat_params = [float(p) for p in params]
            state = _multi_obs_node(*flat_params, shots=shots)
            results.append([complex(val) for val in state])
        return results


__all__ = ["FastBaseEstimator"]
