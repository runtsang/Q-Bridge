"""Quantum estimator with shot‑level simulation and automatic differentiation.

The class now:
* Handles state‑vector or shot‑based evaluation via Aer.
* Supports arbitrary Pauli or sum‑of‑Pauli observables.
* Provides a parameter‑shift gradient routine.
* Caches expectation values for repeated parameter sets.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Dict, List, Tuple, Union

import numpy as np
from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.opflow import PauliSumOp, PauliOp
from qiskit.opflow import StateFn, ExpectationFactory
from qiskit.opflow import AerPauliExpectation, CircuitSampler
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """
    Evaluate expectation values of observables for a parametrised quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to evaluate.
    backend : str | AerSimulator | None
        Backend name or instance.  Defaults to Aer state‑vector simulator.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: str | AerSimulator | None = None,
    ) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend = (
            AerSimulator() if backend is None else backend
        )
        self._cache: Dict[Tuple[float,...], List[complex]] = {}

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Pauli or sum‑of‑Pauli observables.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameter values for the circuit.
        shots : int | None
            If provided, use a shot‑based simulator.  Otherwise use state‑vector.

        Returns
        -------
        List[List[complex]]
            Outer list over parameter sets, inner list over observables.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            key = tuple(values)
            if key in self._cache:
                results.append(self._cache[key])
                continue

            bound_circ = self._bind(values)

            if shots is None:
                state = Statevector.from_instruction(bound_circ)
                row = [state.expectation_value(obs).data for obs in observables]
            else:
                sampler = CircuitSampler(self.backend).convert(
                    StateFn(PauliSumOp.from_list([(obs, 1) for obs in observables]))
                )
                # evaluate with shots
                exp_vals = sampler(bound_circ).eval().data
                row = [exp_vals[i] for i in range(len(observables))]

            self._cache[key] = row
            results.append(row)

        return results

    def gradient(
        self,
        observable: BaseOperator,
        parameter_index: int,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shift: float = np.pi / 2,
    ) -> List[float]:
        """
        Compute the gradient of a single observable w.r.t a single parameter
        using the parameter‑shift rule.

        Parameters
        ----------
        observable : BaseOperator
            The observable to differentiate.
        parameter_index : int
            Index of the parameter to differentiate.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameter values for the circuit.
        shift : float
            Shift value for the parameter‑shift rule.

        Returns
        -------
        List[float]
            Gradient values for each parameter set.
        """
        grads: List[float] = []

        for values in parameter_sets:
            shifted_plus = list(values)
            shifted_minus = list(values)
            shifted_plus[parameter_index] += shift
            shifted_minus[parameter_index] -= shift

            exp_plus = self.evaluate([observable], [shifted_plus], shots=None)[0][0]
            exp_minus = self.evaluate([observable], [shifted_minus], shots=None)[0][0]
            grad = 0.5 * (exp_plus - exp_minus)
            grads.append(grad)

        return grads


__all__ = ["FastBaseEstimator"]
