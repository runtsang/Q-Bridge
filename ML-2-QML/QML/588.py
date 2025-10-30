"""Hybrid estimator that evaluates quantum circuit expectations with parameter‑shift gradients and shot noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators import Operator


class HybridEstimator:
    """Evaluate expectation values of observables for a parametrized quantum circuit with optional shot noise and gradient estimation."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        circuit : QuantumCircuit
            Parametrized quantum circuit to evaluate.
        shots : int, optional
            Number of shots to simulate measurement noise.
        seed : int, optional
            Random seed for reproducibility of shot noise.
        """
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.shots = shots
        self.seed = seed

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Bind a set of parameter values to the circuit."""
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator] | None,
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """
        Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : Iterable[BaseOperator], optional
            Quantum operators to measure. If None, the identity is used.
        parameter_sets : Sequence[Sequence[float]]
            Sequence of parameter vectors to evaluate.

        Returns
        -------
        np.ndarray
            Array of shape (n_params, n_observables) containing expectation values.
        """
        if observables is None or not list(observables):
            observables = [Operator.identity(self._circuit.num_qubits)]
        else:
            observables = list(observables)

        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        results = np.array(results, dtype=complex)
        if self.shots is not None:
            rng = np.random.default_rng(self.seed)
            noise = rng.normal(0, 1 / np.sqrt(self.shots), size=results.shape)
            results = results + noise
        return results

    def gradient(
        self,
        parameter_values: Sequence[float],
        observables: Iterable[BaseOperator] | None = None,
    ) -> np.ndarray:
        """
        Estimate the gradient of expectation values using the parameter‑shift rule.

        Parameters
        ----------
        parameter_values : Sequence[float]
            Current parameter vector.
        observables : Iterable[BaseOperator], optional
            Operators to differentiate. If None, the identity is used.

        Returns
        -------
        np.ndarray
            Gradient array of shape (n_params, n_observables).
        """
        if observables is None or not list(observables):
            observables = [Operator.identity(self._circuit.num_qubits)]
        else:
            observables = list(observables)

        shift = np.pi / 2
        grad: List[np.ndarray] = []

        for i, _ in enumerate(parameter_values):
            values_plus = list(parameter_values)
            values_plus[i] += shift
            state_plus = Statevector.from_instruction(self._bind(values_plus))
            exp_plus = np.array([state_plus.expectation_value(obs) for obs in observables], dtype=complex)

            values_minus = list(parameter_values)
            values_minus[i] -= shift
            state_minus = Statevector.from_instruction(self._bind(values_minus))
            exp_minus = np.array([state_minus.expectation_value(obs) for obs in observables], dtype=complex)

            grad.append((exp_plus - exp_minus) / 2)

        return np.array(grad, dtype=complex)


__all__ = ["HybridEstimator"]
