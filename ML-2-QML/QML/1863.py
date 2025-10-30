"""Fast quantum estimator with shot sampling and parameter‑shift gradients."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[object] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend or Aer.get_backend("statevector_simulator")

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
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Estimator that adds shot‑noise and automatic parameter‑shift gradient support."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[object] = None,
        shots: Optional[int] = None,
    ) -> None:
        super().__init__(circuit, backend=backend)
        self.shots = shots
        if shots is not None:
            # Override backend to a QASM simulator for sampling
            self.backend = Aer.get_backend("qasm_simulator")

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng()
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [rng.normal(float(mean), max(1e-6, 1 / shots)) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def _parameter_shift_expectation(
        self,
        observable: BaseOperator,
        parameter_set: Sequence[float],
        shift: float = np.pi / 2,
    ) -> float:
        """Compute expectation value by evaluating the circuit with +/- shift on each parameter."""
        exp_plus = 0.0
        exp_minus = 0.0
        for i, val in enumerate(parameter_set):
            shifted_plus = list(parameter_set)
            shifted_minus = list(parameter_set)
            shifted_plus[i] += shift
            shifted_minus[i] -= shift
            exp_plus += self.evaluate([observable], [shifted_plus], shots=None)[0][0].real
            exp_minus += self.evaluate([observable], [shifted_minus], shots=None)[0][0].real
        gradient = (exp_plus - exp_minus) / (2 * shift)
        return gradient

    def evaluate_with_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return gradients of each observable w.r.t. the circuit parameters."""
        observables = list(observables)
        grads: List[List[float]] = []
        for params in parameter_sets:
            row_grads: List[float] = []
            for observable in observables:
                grad = self._parameter_shift_expectation(observable, params)
                row_grads.append(grad)
            grads.append(row_grads)
        return grads


__all__ = ["FastBaseEstimator", "FastEstimator"]
