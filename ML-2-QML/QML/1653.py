"""Hybrid estimator that extends FastBaseEstimator with a variational circuit,
automatic parameter optimisation, and shot‑noise simulation."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class QuantumHybridEstimator:
    """
    Evaluates expectation values of observables for a parametrised variational
    circuit.  Supports automatic optimisation of the circuit parameters via
    a classical optimiser and optional shot‑noise simulation using Gaussian
    noise on the expectation values.
    """
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _expectation_statevector(
        self,
        circuit: QuantumCircuit,
        observable: BaseOperator,
    ) -> complex:
        state = Statevector.from_instruction(circuit)
        return state.expectation_value(observable)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.
        If *shots* is provided, Gaussian shot‑noise is added to each mean value.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            row: List[complex] = []
            for observable in observables:
                exp_val = self._expectation_statevector(bound, observable)
                if shots is not None:
                    rng = np.random.default_rng(seed)
                    exp_val = rng.normal(exp_val.real, max(1e-6, 1 / shots)) + 1j * rng.normal(exp_val.imag, max(1e-6, 1 / shots))
                row.append(exp_val)
            results.append(row)
        return results

    def optimise(
        self,
        observable: BaseOperator,
        initial_params: Sequence[float],
        *,
        shots: int = 1024,
        seed: int | None = None,
        max_iter: int = 50,
    ) -> Sequence[float]:
        """
        Optimise the parameters to maximise the expectation value of *observable*
        using a simple gradient‑free optimiser (COBYLA) from scipy.
        """
        from scipy.optimize import minimize

        def objective(params):
            exp = self.evaluate([observable], [params], shots=shots, seed=seed)[0][0]
            return -float(exp)  # negative for maximisation

        result = minimize(
            objective,
            np.array(initial_params),
            method="COBYLA",
            options={"maxiter": max_iter, "disp": False},
        )
        return result.x.tolist()

__all__ = ["QuantumHybridEstimator"]
