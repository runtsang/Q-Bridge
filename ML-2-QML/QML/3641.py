"""Hybrid quantum sampler mirroring the classical network.

The module defines:
* :class:`SamplerQNN` – a parameterized quantum circuit with evaluator.
* :class:`FastEstimator` – extends the base estimator with shot‑noise.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional, List

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler as Sampler

class SamplerQNN:
    """Parameterized quantum circuit that samples from a 2‑qubit state."""
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
    ) -> List[List[complex]]:
        """Compute expectation values for each observable over all parameter sets."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results

class FastEstimator(SamplerQNN):
    """Adds Gaussian shot‑noise to the quantum estimator."""
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        import numpy as np
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [rng.normal(complex(val.real, val.imag), max(1e-6, 1 / shots)) for val in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["SamplerQNN", "FastEstimator"]
