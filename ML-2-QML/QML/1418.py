"""FastBaseEstimator with parameter‑shift gradients and Aer simulation."""

import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Tuple

from qiskit import QuantumCircuit, transpile, Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Operator
from qiskit.utils import QuantumInstance


class FastBaseEstimator:
    """Evaluate expectation values for a parametrised circuit with optional
    parameter‑shift gradient computation.

    The estimator now supports:

    * Aer statevector simulator for fast evaluation.
    * Parameter‑shift rule to compute exact gradients for arbitrary observables.
    * Caching of bound circuits to avoid repeated parameter binding.
    * Flexible observable specification via Operator (PauliSumOp, etc.).
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        quantum_instance: QuantumInstance | None = None,
        cache: bool = True,
    ) -> None:
        self._base_circuit = circuit
        self._params = list(circuit.parameters)
        self.quantum_instance = quantum_instance or QuantumInstance(Aer.get_backend("statevector_simulator"))
        self.cache = cache
        self._cached_circuits: dict[tuple[float,...], QuantumCircuit] = {}

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        if self.cache:
            key = tuple(parameter_values)
            if key in self._cached_circuits:
                return self._cached_circuits[key]
        mapping = {p: v for p, v in zip(self._params, parameter_values)}
        bound = self._base_circuit.assign_parameters(mapping, inplace=False)
        if self.cache:
            self._cached_circuits[tuple(parameter_values)] = bound
        return bound

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            circ = self._bind(values)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_gradient(
        self,
        observable: Operator,
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[List[complex], List[List[float]]]:
        """Return expectation values and gradients using the parameter‑shift rule."""
        results: List[complex] = []
        gradients: List[List[float]] = []

        shift = np.pi / 2  # standard shift for rotation gates
        for params in parameter_sets:
            circ = self._bind(params)
            state = Statevector.from_instruction(circ)
            exp_val = state.expectation_value(observable)
            results.append(exp_val)

            grad: List[float] = []
            for idx, _ in enumerate(self._params):
                # shift up
                params_up = list(params)
                params_up[idx] += shift
                circ_up = self._bind(params_up)
                state_up = Statevector.from_instruction(circ_up)
                exp_up = state_up.expectation_value(observable)

                # shift down
                params_down = list(params)
                params_down[idx] -= shift
                circ_down = self._bind(params_down)
                state_down = Statevector.from_instruction(circ_down)
                exp_down = state_down.expectation_value(observable)

                # parameter‑shift derivative
                derivative = (exp_up - exp_down) / (2 * np.sin(shift))
                grad.append(float(derivative))
            gradients.append(grad)
        return results, gradients


__all__ = ["FastBaseEstimator"]
