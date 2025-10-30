"""Fast estimator for parameterised quantum circuits with shot‑noise and gradient support.

The class FastBaseEstimator accepts a Qiskit QuantumCircuit and
provides batch evaluation of expectation values.  It can run on
Aer simulators with a specified number of shots, caches statevectors
for repeated parameter sets, and offers a parameter‑shift gradient
method.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional, Tuple

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Estimator for expectation values of parameterised quantum circuits.

    Parameters
    ----------
    circuit: QuantumCircuit
        Parameterised circuit to evaluate.  Parameters are bound by
        position.
    backend: str | Aer backend, optional
        Backend to use for simulation.  Defaults to Aer.get_backend('statevector_simulator').
    shots: int | None, optional
        Number of shots to use when a sampler backend is requested.
        If None, the state‑vector simulator is used and results are exact.
    cache: bool, optional
        Cache statevectors for repeated parameter sets.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: str | Aer = "statevector_simulator",
        *,
        shots: int | None = None,
        cache: bool = True,
    ) -> None:
        self._circuit = circuit.copy()
        self._parameters = list(circuit.parameters)
        self.backend = backend if isinstance(backend, str) else backend
        self.shots = shots
        self.cache = cache
        self._state_cache: dict[Tuple[float,...], Statevector] = {}

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _get_statevector(self, params: Sequence[float]) -> Statevector:
        key = tuple(params)
        if self.cache and key in self._state_cache:
            return self._state_cache[key]
        circuit = self._bind(params)
        if self.shots is None:
            state = Statevector.from_instruction(circuit)
        else:
            job = execute(circuit, backend=Aer.get_backend(self.backend), shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            state = Statevector.from_counts(counts)
        if self.cache:
            self._state_cache[key] = state
        return state

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables: Iterable[BaseOperator]
            Operators to measure.
        parameter_sets: Sequence[Sequence[float]]
            Iterable of parameter vectors.

        Returns
        -------
        List[List[complex]]
            Rows correspond to parameter sets, columns to observables.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = self._get_statevector(values)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def gradient(
        self,
        observable: BaseOperator,
        parameter_sets: Sequence[Sequence[float]],
        shift: float = np.pi / 2,
    ) -> List[List[float]]:
        """Compute gradients via parameter‑shift rule.

        Parameters
        ----------
        observable: BaseOperator
            Operator whose expectation value gradient is required.
        parameter_sets: Sequence[Sequence[float]]
            Parameter vectors to evaluate.
        shift: float, optional
            Shift value for the parameter‑shift rule.  Default is π/2.

        Returns
        -------
        List[List[float]]
            Gradient of the observable expectation w.r.t. each parameter.
        """
        grad_results: List[List[float]] = []
        for params in parameter_sets:
            grads = []
            for i, _ in enumerate(params):
                plus = list(params)
                minus = list(params)
                plus[i] += shift
                minus[i] -= shift
                plus_state = self._get_statevector(plus)
                minus_state = self._get_statevector(minus)
                value_plus = plus_state.expectation_value(observable)
                value_minus = minus_state.expectation_value(observable)
                grads.append(float((value_plus - value_minus) / (2 * np.sin(shift))))
            grad_results.append(grads)
        return grad_results


__all__ = ["FastBaseEstimator"]
