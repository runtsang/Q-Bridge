"""Hybrid estimator for Qiskit quantum circuits.

The class wraps a :class:`qiskit.circuit.QuantumCircuit` and evaluates the
expectation value of a list of :class:`qiskit.quantum_info.operators.base_operator.BaseOperator`
for a batch of parameter sets.  An optional ``shots`` argument adds Gaussian
noise to the exact state‑vector expectation to emulate shot noise."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class FastBaseEstimator:
    """Evaluate expectation values of a parametrized Qiskit circuit."""

    def __init__(self, circuit: QuantumCircuit) -> None:
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
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Return expectation values for each parameter set and observable.

        Parameters
        ----------
        observables:
            Iterable of BaseOperator objects from Qiskit.
        parameter_sets:
            Iterable of parameter sequences to bind to the circuit.
        shots:
            If provided, Gaussian noise with variance ``1/shots`` is added to
            the exact Statevector expectation to emulate shot noise.
        seed:
            Random seed for reproducibility of the noise.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            state = Statevector.from_instruction(self._bind(params))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                complex(rng.normal(val.real, max(1e-6, 1 / shots)), 0.0)
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator"]
