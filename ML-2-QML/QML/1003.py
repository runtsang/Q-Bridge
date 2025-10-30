"""FastBaseEstimator for quantum circuits with advanced evaluation.

The estimator supports batched expectation value evaluation for
parametrized circuits on Qiskit Aer simulators, optional shot noise,
and a parameter‑shift gradient routine.  It retains the original API
while adding richer functionality for research workflows.
"""

from __future__ import annotations

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List, Union


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The parametrized quantum circuit to evaluate.
    shots : int | None, optional
        Number of measurement shots to emulate.  If ``None`` the
        exact statevector expectation is returned.
    seed : int | None, optional
        Random seed for shot‑noise simulation.
    """

    def __init__(self, circuit: QuantumCircuit, shots: int | None = None, seed: int | None = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.shots = shots
        self.rng = np.random.default_rng(seed)
        # Choose backend: statevector for exact, aer for shot simulation
        self.backend = Aer.get_backend("aer_simulator_statevector") if shots is None else Aer.get_backend("aer_simulator")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of operators (e.g., Pauli operators) whose
            expectation values are to be computed.
        parameter_sets
            Iterable of parameter sequences.  Each sequence is bound to
            the circuit before evaluation.

        Returns
        -------
        List[List[complex]]
            Outer list over parameter sets, inner list over observables.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]

            if self.shots is not None:
                # Add shot‑noise by sampling from a normal distribution
                noisy_row = []
                for val in row:
                    mean = float(val)
                    var = (1 - mean**2) / self.shots
                    noisy_val = self.rng.normal(mean, np.sqrt(max(var, 1e-12)))
                    noisy_row.append(noisy_val)
                row = noisy_row

            results.append(row)

        return results

    def gradient(self, observable: BaseOperator, parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """Compute the gradient of a single observable w.r.t. circuit parameters.

        Uses the parameter‑shift rule with a shift of π/2.

        Parameters
        ----------
        observable
            The operator whose gradient is desired.
        parameter_sets
            Iterable of parameter sequences.

        Returns
        -------
        List[List[float]]
            Outer list over parameter sets, inner list over parameters.
        """
        shift = np.pi / 2
        grads: List[List[float]] = []

        for values in parameter_sets:
            grad_row: List[float] = []
            for i, _ in enumerate(self._parameters):
                shifted_plus = list(values)
                shifted_plus[i] += shift
                state_plus = Statevector.from_instruction(self._bind(shifted_plus))
                exp_plus = state_plus.expectation_value(observable)

                shifted_minus = list(values)
                shifted_minus[i] -= shift
                state_minus = Statevector.from_instruction(self._bind(shifted_minus))
                exp_minus = state_minus.expectation_value(observable)

                grad = 0.5 * (exp_plus - exp_minus)
                grad_row.append(float(grad))
            grads.append(grad_row)

        return grads
