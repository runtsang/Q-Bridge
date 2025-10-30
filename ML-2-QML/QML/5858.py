"""Hybrid quantum estimator that evaluates expectation values of parametrised circuits.

This module extends the original FastBaseEstimator with a robust interface for
Qiskit circuits, supporting arbitrary observables, parameter binding, and
shot‑noise emulation.  It can be used directly as a drop‑in replacement for the
classical version when the model is a variational quantum circuit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import PauliZ
from qiskit.quantum_info.operators.base_operator import BaseOperator


class HybridBaseEstimator:
    """
    Evaluate a Qiskit quantum circuit for batches of parameters and observables.

    Parameters
    ----------
    circuit : QuantumCircuit
        A parameterised quantum circuit that accepts a set of variational angles.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """
        Return a new circuit with parameters bound to the supplied values.
        """
        if len(parameter_values)!= len(self._parameters):
            raise ValueError(
                "Parameter count mismatch for bound circuit. "
                f"Expected {len(self._parameters)} but got {len(parameter_values)}."
            )
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables
            Qiskit BaseOperator instances (e.g. PauliZ, H, custom operators).
            If ``None`` a single Pauli‑Z operator is used.
        parameter_sets
            2‑D sequence of parameter values corresponding to the circuit's
            parameters.

        Returns
        -------
        List[List[complex]]
            A list of rows, one per parameter set, each row containing the
            expectation values of the supplied observables.
        """
        if parameter_sets is None:
            raise ValueError("parameter_sets must be provided")
        observables = list(observables or [PauliZ()])
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_noisy(
        self,
        observables: Iterable[BaseOperator] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Same as :meth:`evaluate` but adds Gaussian shot noise to each expectation value.

        Parameters
        ----------
        shots
            Number of classical shots to emulate.  If ``None`` the method simply
            forwards to :meth:`evaluate`.
        seed
            Random seed for reproducibility.
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                rng.normal(val.real, max(1e-6, 1 / shots)) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


class HybridEstimator(HybridBaseEstimator):
    """
    Convenience subclass that exposes the noisy evaluation as ``evaluate``.
    """

    def evaluate(
        self,
        observables: Iterable[BaseOperator] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        return self.evaluate_noisy(observables, parameter_sets, shots=shots, seed=seed)


__all__ = ["HybridBaseEstimator", "HybridEstimator"]
