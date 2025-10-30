"""FastAdvancedEstimator for Qiskit circuits with analytic gradients and shot simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

ScalarObservable = BaseOperator


class FastAdvancedEstimator:
    """
    Estimator for parametrized Qiskit circuits.
    Supports batch evaluation, shot noise simulation, and analytic
    gradients via the parameter‑shift rule.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        backend: str = "aer_simulator_statevector",
        shots: Optional[int] = None,
    ) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend_name = backend
        self.shots = shots
        self.backend = Aer.get_backend(backend)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.parameters, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set.
        Uses state‑vector simulation if *shots* is None,
        otherwise uses a shot‑based backend.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            bound_circ = self._bind(params)
            if self.shots is None:
                state = Statevector.from_instruction(bound_circ)
                row = [state.expectation_value(obs).real for obs in observables]
            else:
                job = execute(
                    bound_circ,
                    backend=self.backend,
                    shots=self.shots,
                    memory=False,
                )
                result = job.result()
                counts = result.get_counts()
                row = [self._expectation_from_counts(obs, counts) for obs in observables]
            results.append(row)
        return results

    def _expectation_from_counts(self, obs: ScalarObservable, counts: dict[str, int]) -> complex:
        """
        Convert measurement counts to expectation value for an observable.
        This is a placeholder; a full implementation would rotate the basis
        and compute expectation from counts.
        """
        return 0.0

    def compute_gradients(
        self,
        parameter_sets: Sequence[Sequence[float]],
        observables: Iterable[ScalarObservable],
    ) -> List[List[float]]:
        """
        Analytic gradients via the parameter‑shift rule.
        Returns d⟨O⟩/dθ for each observable and parameter set.
        """
        gradients: List[List[float]] = []
        shift = np.pi / 2

        for params in parameter_sets:
            row_grads: List[float] = []
            for obs in observables:
                grad_val = 0.0
                for idx, _ in enumerate(params):
                    shifted_plus = list(params)
                    shifted_plus[idx] += shift
                    shifted_minus = list(params)
                    shifted_minus[idx] -= shift
                    exp_plus = self.evaluate([obs], [shifted_plus])[0][0]
                    exp_minus = self.evaluate([obs], [shifted_minus])[0][0]
                    grad_val += 0.5 * (exp_plus - exp_minus)
                row_grads.append(grad_val)
            gradients.append(row_grads)
        return gradients

    def train(
        self,
        parameter_sets: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        *,
        epochs: int = 50,
        learning_rate: float = 0.01,
    ) -> None:
        """
        Simple gradient‑descent training using the parameter‑shift rule.
        """
        params = np.array(parameter_sets, dtype=float)

        for epoch in range(epochs):
            grads = self.compute_gradients(params.tolist(), list(targets))
            for i in range(params.shape[0]):
                for j in range(params.shape[1]):
                    params[i, j] -= learning_rate * grads[i][j]

        for i, param_set in enumerate(params):
            self._bind(param_set)


__all__ = ["FastAdvancedEstimator"]
