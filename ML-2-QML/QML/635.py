"""Quantum estimator with optional shot‑based evaluation and backend selection."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, BaseOperator
from qiskit.providers.aer import Aer
from qiskit.result import Result


class FastBaseEstimatorGen:
    """Evaluate expectation values of observables for a parameterised quantum circuit.

    Parameters
    ----------
    circuit
        A :class:`~qiskit.circuit.QuantumCircuit` with symbolic parameters.
    backend
        Optional :class:`~qiskit.providers.backend.Backend`.  When ``None`` a
        state‑vector simulator is used.  Any backend that supports
        ``run(..., shots=…)`` can be supplied.
    """

    def __init__(self, circuit: QuantumCircuit, backend: Optional[object] = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._backend = backend or Aer.get_backend("statevector_simulator")

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
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of :class:`~qiskit.quantum_info.operators.base_operator.BaseOperator`
            representing the measurement operators.
        parameter_sets
            Sequence of parameter vectors.
        shots
            If provided, the expectation values are perturbed with Gaussian
            noise having variance ``1/shots`` to emulate finite‑shot sampling.
        seed
            Seed for the Gaussian noise generator.

        Returns
        -------
        List[List[complex]]
            A matrix of shape ``(len(parameter_sets), len(observables))``.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        rng = np.random.default_rng(seed)

        for values in parameter_sets:
            bound = self._bind(values)

            if shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Deterministic expectation via statevector
                state = Statevector.from_instruction(bound)
                true_vals = [state.expectation_value(obs) for obs in observables]
                # Add shot‑noise
                row = [
                    float(rng.normal(float(val), max(1e-6, 1 / shots)))
                    for val in true_vals
                ]
            results.append(row)

        return results


__all__ = ["FastBaseEstimatorGen"]
