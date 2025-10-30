"""Extended FastBaseEstimator for Qiskit circuits with batched expectation evaluation, caching, and simulated shot noise."""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_aer import AerSimulator

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit.
    Supports backend selection, batched evaluation, caching, and optional shot noise simulation.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[AerSimulator] = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._backend = backend or AerSimulator()
        self._shots = shots
        self._seed = seed
        self._cache = {}

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _expectation_batch(
        self, parameter_sets: Sequence[Sequence[float]], observables: Iterable[BaseOperator]
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        observables = list(observables)
        if shots is None:
            shots = self._shots
        if shots is None:
            # use statevector simulation
            cache_key = tuple(tuple(p) for p in parameter_sets)
            if cache_key in self._cache:
                return self._cache[cache_key]
            results = self._expectation_batch(parameter_sets, observables)
            self._cache[cache_key] = results
            return results
        # shot simulation using AerSimulator
        rng = np.random.default_rng(seed or self._seed)
        results = []
        for values in parameter_sets:
            bound_circ = self._bind(values)
            job = self._backend.run(bound_circ, shots=shots, seed_simulator=seed)
            counts = job.result().get_counts()
            row = []
            for obs in observables:
                exp = 0.0
                for bitstring, count in counts.items():
                    exp += obs.expectation_value(Statevector.from_label(bitstring)) * count
                exp /= shots
                row.append(exp)
            results.append(row)
        return results


__all__ = ["FastBaseEstimator"]
