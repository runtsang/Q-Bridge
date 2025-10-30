"""Quantum estimator with support for parameter‑shift gradients and shot‑noise simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.opflow import PauliSumOp, StateFn, AerPauliExpectation, PauliExpectation, ExpectationFactory
from qiskit.opflow import PrimitiveOp

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit.

    Supports both state‑vector and shot‑based simulation and provides a
    parameter‑shift gradient routine.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrized quantum circuit.
    backend_name : str | None, optional
        Name of the Qiskit Aer backend to use.  Defaults to
        ``statevector_simulator`` for exact evaluation or
        ``qasm_simulator`` for shot‑based simulation.
    shots : int | None, optional
        Number of shots for a shot‑based simulation.  ``None`` triggers
        exact state‑vector evaluation.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        backend_name: str | None = None,
        shots: int | None = None,
    ) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.shots = shots
        backend_name = backend_name or (
            "statevector_simulator" if shots is None else "qasm_simulator"
        )
        self.backend = Aer.get_backend(backend_name)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def _expectation_statevector(
        self, state: Statevector, observable: BaseOperator
    ) -> complex:
        return state.expectation_value(observable)

    def _expectation_shots(
        self, counts: dict[str, int], observable: BaseOperator
    ) -> complex:
        """Compute expectation of a PauliSumOp from shot counts."""
        if not isinstance(observable, PauliSumOp):
            raise TypeError("Shot‑based evaluation requires PauliSumOp observables.")
        expectation = AerPauliExpectation()
        op = PrimitiveOp(observable)
        exp_val = expectation.convert(StateFn(op)).eval()
        return complex(exp_val)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        if self.shots is None:
            # Exact state‑vector evaluation
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [self._expectation_statevector(state, obs) for obs in observables]
                results.append(row)
        else:
            # Shot‑based evaluation
            for values in parameter_sets:
                bound = self._bind(values)
                job = execute(
                    bound,
                    backend=self.backend,
                    shots=self.shots,
                    parameter_binds=[dict(zip(self.parameters, values))],
                )
                result = job.result()
                counts = result.get_counts()
                row = [
                    self._expectation_shots(counts, obs) for obs in observables
                ]
                results.append(row)

        return results

    def gradient(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shift: float = np.pi / 2,
    ) -> List[List[complex]]:
        """Compute gradients of expectation values w.r.t. circuit parameters using
        the parameter‑shift rule.  Only exact state‑vector evaluation is
        supported for gradients."""
        if self.shots is not None:
            raise NotImplementedError(
                "Gradient evaluation is only available for exact state‑vector simulation."
            )

        grad_results: List[List[complex]] = []
        for values in parameter_sets:
            grad_row: List[complex] = []
            for obs in observables:
                grad = 0.0
                for idx, _ in enumerate(self.parameters):
                    plus = list(values)
                    minus = list(values)
                    plus[idx] += shift
                    minus[idx] -= shift
                    state_plus = Statevector.from_instruction(self._bind(plus))
                    state_minus = Statevector.from_instruction(self._bind(minus))
                    exp_plus = self._expectation_statevector(state_plus, obs)
                    exp_minus = self._expectation_statevector(state_minus, obs)
                    grad += 0.5 * (exp_plus - exp_minus)
                grad_row.append(grad)
            grad_results.append(grad_row)

        return grad_results


__all__ = ["FastBaseEstimator"]
