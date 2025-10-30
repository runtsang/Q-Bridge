"""Enhanced quantum estimator with shot noise, state‑vector simulation and
parameter‑shift gradients."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Base class to evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Extension of FastBaseEstimator that adds shot‑noise simulation and
    analytical gradients via the parameter‑shift rule."""
    def __init__(self, circuit: QuantumCircuit, shots: int | None = None) -> None:
        super().__init__(circuit)
        self._shots = shots
        if shots is not None:
            self._sim = Aer.get_backend("qasm_simulator")
        else:
            self._sim = Aer.get_backend("statevector_simulator")

    def _simulate(self, bound_circuit: QuantumCircuit) -> Statevector:
        if self._shots is None:
            return Statevector.from_instruction(bound_circuit)
        result = self._sim.run(bound_circuit, shots=self._shots).result()
        state = Statevector(result.get_statevector(bound_circuit))
        return state

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Override to include optional shot noise."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            state = self._simulate(bound)
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results

    def evaluate_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[List[List[complex]], List[List[List[complex]]]]:
        """
        Compute analytical gradients of each observable with respect to every circuit
        parameter using the parameter‑shift rule.
        Returns a tuple (values, gradients) where gradients[idx][obs][param] is the gradient
        of observable `obs` for parameter set `idx` with respect to parameter `param`.
        """
        observables = list(observables)
        values: List[List[complex]] = []
        gradients: List[List[List[complex]]] = []

        shift = np.pi / 2

        for values_set in parameter_sets:
            base_row = self.evaluate(observables, [values_set])[0]
            grad_row: List[List[complex]] = []

            for _obs in observables:
                grad_per_param: List[complex] = []

                for i, _ in enumerate(self._parameters):
                    shifted_plus = list(values_set)
                    shifted_minus = list(values_set)
                    shifted_plus[i] += shift
                    shifted_minus[i] -= shift

                    plus_exp = self.evaluate(observables, [shifted_plus])[0][_obs]
                    minus_exp = self.evaluate(observables, [shifted_minus])[0][_obs]
                    gradient = (plus_exp - minus_exp) * 0.5
                    grad_per_param.append(gradient)

                grad_row.append(grad_per_param)

            values.append(base_row)
            gradients.append(grad_row)

        return values, gradients


__all__ = ["FastBaseEstimator", "FastEstimator"]
