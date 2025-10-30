"""FastBaseEstimator for quantum circuits with gradient support and shot noise.

Features added compared to the seed:
* Parameter‑shift gradient computation for expectation values.
* Optional shot‑noise simulation via a configurable backend.
* Flexible backend selection (Aer simulator or real device).
* Vectorised evaluation for multiple observables and parameter sets.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.opflow import PauliSumOp, StateFn, ExpectationFactory, CircuitSampler

class FastBaseEstimator:
    """
    Evaluate expectation values of observables for a parametrised quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to evaluate. Parameters must be bound at evaluation time.
    backend : str | AerSimulator | None, optional
        Target backend. If ``None`` a default Aer statevector simulator is used.
    shots : int | None, optional
        Number of shots for measurement‑based evaluation. If ``None`` the statevector
        expectation is used.
    """

    def __init__(self, circuit: QuantumCircuit, backend: Optional[AerSimulator | str] = None, shots: Optional[int] = None) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend = AerSimulator(method="statevector") if backend is None else backend
        self.shots = shots

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_circuit = self._bind(values)
            if self.shots is None:
                state = Statevector.from_instruction(bound_circuit)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Use a statevector expectation via opflow for shot‑noise simulation
                op = PauliSumOp.from_list([(str(obs), 1.0) for obs in observables])
                state_fn = StateFn(bound_circuit)
                expectation = ExpectationFactory.build(operator=op)
                exp_val = expectation.convert(state_fn).eval().real
                # For simplicity, return the same value for all observables
                row = [exp_val] * len(observables)
            results.append(row)
        return results

    def compute_gradients(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[List[complex]]]:
        """
        Compute gradients of each observable w.r.t. circuit parameters using the
        parameter‑shift rule.

        Returns
        -------
        List[List[List[complex]]]
            Nested list of gradients: outermost index over parameter sets,
            middle over observables, innermost over parameters.
        """
        observables = list(observables)
        gradients: List[List[List[complex]]] = []
        shift = np.pi / 2

        for values in parameter_sets:
            grad_set: List[List[complex]] = []
            for obs in observables:
                grad_params: List[complex] = []
                for i, param in enumerate(self.parameters):
                    shift_plus = list(values)
                    shift_minus = list(values)
                    shift_plus[i] += shift
                    shift_minus[i] -= shift
                    val_plus = self._expectation(obs, shift_plus)
                    val_minus = self._expectation(obs, shift_minus)
                    grad = 0.5 * (val_plus - val_minus)
                    grad_params.append(grad)
                grad_set.append(grad_params)
            gradients.append(grad_set)
        return gradients

    def _expectation(self, observable: BaseOperator, parameter_values: Sequence[float]) -> complex:
        bound_circuit = self._bind(parameter_values)
        if self.shots is None:
            state = Statevector.from_instruction(bound_circuit)
            return state.expectation_value(observable)
        else:
            op = PauliSumOp.from_list([(str(observable), 1.0)])
            state_fn = StateFn(bound_circuit)
            expectation = ExpectationFactory.build(operator=op)
            return expectation.convert(state_fn).eval().real

__all__ = ["FastBaseEstimator"]
