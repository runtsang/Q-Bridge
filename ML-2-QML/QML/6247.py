"""Hybrid quantum estimator that evaluates a parametrized circuit and an optional sampler."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List
import numpy as np

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Sampler as QSampler


def _build_sampler_circuit() -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Builds a simple 2‑qubit sampler QNN circuit."""
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    return qc, inputs, weights


class FastHybridQuantumEstimator:
    """Evaluate expectation values of a parametrized circuit and an optional sampler."""

    def __init__(self, circuit: QuantumCircuit, sampler: QSampler | None = None) -> None:
        self.circuit = circuit
        self._parameters = list(circuit.parameters)
        self.sampler = sampler

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Same as evaluate but adds Poisson shot‑noise using the supplied sampler."""
        if self.sampler is None:
            raise RuntimeError("No sampler provided for shot‑noise evaluation.")
        results: List[List[complex]] = []
        rng = np.random.default_rng(seed)
        for values in parameter_sets:
            # Sample probabilities from the circuit
            probs = self.sampler.run(self._bind(values)).probabilities()
            # Poisson sampling of counts from expected probabilities
            counts = rng.poisson(lam=shots, size=len(probs))
            # Build a statevector from the most likely outcome
            est_state = Statevector.from_label(counts.argmax())
            row = [est_state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


def make_hybrid_estimator(shots: int | None = None) -> FastHybridQuantumEstimator:
    """Convenience factory that bundles the sampler circuit."""
    qc, _, _ = _build_sampler_circuit()
    sampler = QSampler() if shots else None
    return FastHybridQuantumEstimator(circuit=qc, sampler=sampler)


__all__ = ["FastHybridQuantumEstimator", "make_hybrid_estimator"]
