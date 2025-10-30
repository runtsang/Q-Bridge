"""Hybrid fast estimator for Qiskit circuits with optional shot noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class HybridFastEstimator:
    """
    Lightweight estimator for parametrised Qiskit circuits.

    Parameters
    ----------
    circuit : QuantumCircuit
        A circuit that may contain symbolic parameters.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each observable and parameter set.

        Parameters
        ----------
        observables
            Iterable of BaseOperator objects.
        parameter_sets
            Iterable of parameter vectors.
        shots
            If supplied, simulate measurement shot noise by re‑sampling the
            statevector expectation with a Gaussian of variance 1/shots.
        seed
            Random seed for reproducibility.

        Returns
        -------
        List[List[complex]]
            Nested list of expectation values.
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
                rng.normal(complex(val.real, val.imag), max(1e-6, 1 / shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

    # ------------------------------------------------------------------
    # Convenience wrappers for the quantum FCL example
    # ------------------------------------------------------------------
    def run_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the FCL quantum circuit when it implements a ``run`` method.

        Parameters
        ----------
        thetas
            Iterable of parameters for the circuit.

        Returns
        -------
        np.ndarray
            The expectation value output by the circuit.
        """
        if not hasattr(self._circuit, "run"):
            raise AttributeError("Circuit does not expose a `run` method.")
        return self._circuit.run(thetas)

__all__ = ["HybridFastEstimator"]
