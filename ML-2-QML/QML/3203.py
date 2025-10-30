"""Fast hybrid estimator for quantum circuits with optional shot noise.

The implementation follows the classical API but operates on
Qiskit Statevector instances.  An optional ``shots`` argument
adds Gaussian shot noise to the expectation values, mimicking
finite‑sample measurement statistics.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastHybridEstimator:
    """Evaluate a parametrised Qiskit circuit for many parameter sets."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, param_vals: Sequence[float]) -> QuantumCircuit:
        """Return a new circuit with parameters bound."""
        if len(param_vals)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, param_vals))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set.

        If ``shots`` is provided, Gaussian noise with variance
        1/shots is added to each expectation value to emulate
        finite‑shot statistics.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Operators whose expectation values are desired.
        parameter_sets : sequence of sequences
            Parameter values to bind to the circuit.
        shots : int, optional
            Number of measurement shots to simulate.
        seed : int, optional
            Random seed for the Gaussian noise.

        Returns
        -------
        List[List[complex]]
            Outer dimension matches ``parameter_sets``; inner dimension
            matches ``observables``.
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
                val + rng.normal(0, max(1e-6, 1 / shots)) for val in row
            ]
            noisy.append(noisy_row)

        return noisy


__all__ = ["FastHybridEstimator"]
