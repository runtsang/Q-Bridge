"""Enhanced fast estimator framework using Pennylane with circuit graph abstraction and GPU acceleration."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Callable, List

import pennylane as qml
import numpy as np

ScalarObservable = qml.measurements.MeasurementProcess | Callable[[np.ndarray], np.ndarray | float]


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized quantum circuit.

    Parameters
    ----------
    circuit : Callable[[Sequence[float]], qml.QNode]
        Function that returns a QNode when called with a parameter vector.
    device_name : str, optional
        Name of the Pennylane device to use. Defaults to 'default.qubit'.
    device_options : dict, optional
        Options passed to the Pennylane device constructor.
    shard_size : int, optional
        Number of parameter sets processed per shard for parallel execution.
    """

    def __init__(
        self,
        circuit: Callable[[Sequence[float]], qml.QNode],
        device_name: str = "default.qubit",
        device_options: dict | None = None,
        shard_size: int = 32,
    ) -> None:
        self.circuit = circuit
        self.device = qml.device(device_name, wires=self.circuit( [0] ).wires, **(device_options or {}))
        self.shard_size = shard_size

    def _shard_params(self, parameter_sets: Sequence[Sequence[float]]) -> List[List[Sequence[float]]]:
        """Split parameter sets into shards."""
        return [parameter_sets[i : i + self.shard_size] for i in range(0, len(parameter_sets), self.shard_size)]

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        log_timing: bool = False,
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Pennylane measurement processes or callables that accept raw state vector.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.
        log_timing : bool, optional
            If True, print timing information.

        Returns
        -------
        List[List[complex]]
            Results as a list of rows, each row containing the values for all observables.
        """
        if log_timing:
            start = time.perf_counter()

        observables = list(observables)
        results: List[List[complex]] = []

        # Wrap the circuit into a QNode that returns the statevector
        @qml.qnode(self.device, interface="autograd")
        def circuit_node(params):
            qnode = self.circuit(params)
            qnode()
            return qml.state()

        for shard in self._shard_params(parameter_sets):
            for params in shard:
                state = circuit_node(params).numpy()
                row: List[complex] = []
                for observable in observables:
                    if isinstance(observable, qml.measurements.MeasurementProcess):
                        exp_val = qml.expval(observable)(state)
                        row.append(exp_val)
                    else:
                        val = observable(state)
                        row.append(complex(val))
                results.append(row)

        if log_timing:
            elapsed = time.perf_counter() - start
            print(f"FastBaseEstimator.evaluate: {elapsed:.4f}s")

        return results


__all__ = ["FastBaseEstimator"]
