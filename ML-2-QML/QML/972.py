"""FastBaseEstimator for quantum circuits with gradient support.

The implementation builds on Qiskit and optionally uses the Aer
simulator.  It can evaluate expectation values, simulate shot noise,
and compute parameter‑shift gradients for a list of observables.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator

ScalarExpectation = BaseOperator  # alias for readability


def _bind(circuit: QuantumCircuit, parameters: Sequence[float]) -> QuantumCircuit:
    """Return a new circuit with parameters bound."""
    if len(parameters)!= len(circuit.parameters):
        raise ValueError("Parameter count mismatch for bound circuit.")
    mapping = dict(zip(circuit.parameters, parameters))
    return circuit.assign_parameters(mapping, inplace=False)


class FastBaseEstimator:
    """Estimator for parametrized quantum circuits.

    Supports state‑vector evaluation, Aer simulation with shot noise,
    and parameter‑shift gradient computation.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: str = "statevector",
        shots: int | None = None,
    ) -> None:
        self.circuit = circuit
        self.backend = backend
        self.shots = shots
        if backend == "aer":
            self.sim = AerSimulator()
        else:
            self.sim = None

    def _prepare_state(self, params: Sequence[float]) -> Statevector:
        bound = _bind(self.circuit, params)
        if self.backend == "statevector":
            return Statevector.from_instruction(bound)
        else:  # Aer simulation
            result = self.sim.run(bound, shots=self.shots).result()
            statevec = result.get_statevector()
            return Statevector(statevec)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each observable and parameter set.

        Parameters
        ----------
        observables: iterable of BaseOperator
            Operators whose expectation values are desired.
        parameter_sets: sequence of parameter vectors
            Each vector is bound to the circuit.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            state = self._prepare_state(params)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[List[complex]]:
        """Same as :meth:`evaluate` but forces a shot‑based simulation."""
        if shots is None:
            shots = self.shots or 1024
        self.shots = shots
        self.backend = "aer"
        self.sim = AerSimulator()
        return self.evaluate(observables, parameter_sets)

    def _parameter_shift(
        self,
        observable: BaseOperator,
        params: Sequence[float],
        shift: float = np.pi / 2,
    ) -> List[complex]:
        """Compute the derivative of an expectation value w.r.t. each parameter."""
        grads = []
        for i in range(len(params)):
            plus = list(params)
            minus = list(params)
            plus[i] += shift
            minus[i] -= shift
            state_plus = self._prepare_state(plus)
            state_minus = self._prepare_state(minus)
            exp_plus = state_plus.expectation_value(observable)
            exp_minus = state_minus.expectation_value(observable)
            grad = (exp_plus - exp_minus) / (2 * np.sin(shift))
            grads.append(grad)
        return grads

    def evaluate_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[complex]]]:
        """
        Return parameter‑shift gradients for each observable and parameter set.

        The output shape is ``[n_sets][n_observables][n_params]``.
        """
        observables = list(observables)
        grads_all: List[List[List[complex]]] = []

        for params in parameter_sets:
            grads_set: List[List[complex]] = []
            for obs in observables:
                grads_set.append(self._parameter_shift(obs, params))
            grads_all.append(grads_set)
        return grads_all


__all__ = ["FastBaseEstimator"]
