"""Hybrid estimator that evaluates a Qiskit circuit.

The implementation mirrors the classical counterpart but operates on a
parameter‑dependent quantum circuit.  It returns a matrix of complex
expectation values for a list of `qiskit.quantum_info.operators.base_operator.BaseOperator`
observables.  An optional `shots` argument can be supplied to add Gaussian
shot‑noise to the exact expectation values, emulating a finite‑sample
measurement.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Callable

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class HybridEstimator:
    """Evaluate a parameterised Qiskit circuit for a set of observables."""
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
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Return expectation values for each parameter set and observable.

        Parameters
        ----------
        observables:
            Iterable of BaseOperator objects whose expectation values are
            evaluated on the final state.
        parameter_sets:
            Sequence of parameter tuples; each tuple is bound to the circuit.
        shots:
            If provided, Gaussian noise with variance 1/shots is added to each
            expectation value to emulate shot noise.
        seed:
            Seed for the pseudo‑random number generator used in shot noise.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                complex(rng.normal(val.real, max(1e-6, 1 / shots)),
                        rng.normal(val.imag, max(1e-6, 1 / shots)))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridEstimator"]
