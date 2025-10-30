"""Unified estimator for Qiskit quantum circuits.

The estimator mirrors the PyTorch counterpart but operates on a
``QuantumCircuit``.  It evaluates expectation values of a set of
``BaseOperator`` observables for each parameter set and optionally
injects shot‑noise to emulate realistic measurements.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Sequence as Seq

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class UnifiedBaseEstimator:
    """Estimator for quantum circuits.

    Parameters
    ----------
    circuit:
        A Qiskit ``QuantumCircuit`` that may have symbolic parameters.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit
        self.params = list(circuit.parameters)

    def _bind(self, values: Seq[float]) -> QuantumCircuit:
        """Return a new circuit with the symbolic parameters bound to ``values``."""
        if len(values)!= len(self.params):
            raise ValueError(
                f"Expected {len(self.params)} parameter values, got {len(values)}."
            )
        mapping = dict(zip(self.params, values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each observable and parameter set.

        Parameters
        ----------
        observables:
            Iterable of ``BaseOperator`` instances.
        parameter_sets:
            Iterable of parameter sequences; each sequence is bound to the
            circuit and used to generate a statevector.
        shots:
            If provided, Gaussian noise with ``var=1/shots`` is added to each
            expectation value to emulate measurement statistics.
        seed:
            Random seed for reproducibility of synthetic shot noise.
        Returns
        -------
        List[List[complex]]:
            A list of rows, one per parameter set.  Each row contains the
            expectation value of every observable in the order supplied.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            circ = self._bind(values)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                rng.normal(float(v.real), max(1e-6, 1 / shots))
                + 1j * rng.normal(float(v.imag), max(1e-6, 1 / shots))
                for v in row
            ]
            noisy.append(noisy_row)
        return noisy

__all__ = ["UnifiedBaseEstimator"]
