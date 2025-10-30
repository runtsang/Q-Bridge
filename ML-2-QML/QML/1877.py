"""Hybrid estimator that evaluates a variational quantum circuit with an optional classical
feature extractor.  The estimator accepts a Qiskit QuantumCircuit and a PyTorch model
that can be used to transform raw parameters into the circuit’s parameters.

The public API is identical to the classical variant – ``evaluate`` takes an iterable of
quantum observables and a sequence of raw parameter sets.  If a feature extractor is
provided, each raw set is first mapped to the circuit’s parameters.  Optional
``shots`` adds Gaussian noise to the expectation values to emulate measurement
statistics.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastHybridEstimator:
    """Evaluate a quantum circuit with an optional classical feature extractor.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterised variational circuit whose parameters will be bound
        to each evaluation.
    feature_extractor : Callable[[Sequence[float]], Sequence[float]] | None, optional
        A pure‑function or PyTorch model that maps raw parameters to the
        circuit parameters.  If ``None`` the raw values are used directly.
    shots : int | None, optional
        Number of simulated shots.  When set, Gaussian noise with standard
        deviation ``1 / sqrt(shots)`` is added to each expectation value.
    seed : int | None, optional
        Random seed for the noise generator.
    """

    __slots__ = ("_circuit", "_parameters", "feature_extractor", "shots", "seed", "rng")

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        feature_extractor: Callable[[Sequence[float]], Sequence[float]] | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.feature_extractor = feature_extractor
        self.shots = shots
        self.seed = seed
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a new circuit with parameters bound."""
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return a 2‑D list of expectation values for each parameter set.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Quantum observables to evaluate on the output state.
        parameter_sets : sequence of parameter sequences
            Raw parameters that may be transformed by ``feature_extractor``.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for raw_values in parameter_sets:
            # Optional classical preprocessing
            values = (
                self.feature_extractor(raw_values) if self.feature_extractor else raw_values
            )
            circ = self._bind(values)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]

            if self.shots is not None:
                # Add Gaussian shot noise to each expectation value
                noisy_row = [
                    val.real + self.rng.normal(0, 1 / np.sqrt(self.shots)) for val in row
                ]
                row = [complex(v) for v in noisy_row]

            results.append(row)

        return results


__all__ = ["FastHybridEstimator"]
