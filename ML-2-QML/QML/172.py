"""Hybrid estimator that evaluates a parametrised quantum circuit with optional shot noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit import transpile
from qiskit.providers.aer import AerSimulator
from qiskit.opflow import PauliExpectation, PauliSumOp, StateFn, CircuitSampler

ScalarObservable = BaseOperator  # alias for clarity


class FastHybridEstimator:
    """Quantum estimator that evaluates expectation values of observables for a parametrised circuit.

    Parameters
    ----------
    circuit
        A Qiskit ``QuantumCircuit`` with symbolic parameters.
    backend
        Simulation backend to use: ``'qiskit'`` for exact Statevector simulation,
        ``'qasm'`` for shot‑based simulation with Aer.
    """
    def __init__(self, circuit: QuantumCircuit, backend: str = "qiskit"):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        if backend not in ("qiskit", "qasm"):
            raise ValueError("backend must be 'qiskit' or 'qasm'")
        if backend == "qasm" and not hasattr(AerSimulator, "__call__"):
            raise ImportError("AerSimulator required for shot simulation")
        self.backend = backend

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _evaluate_exact(self, observables: Iterable[BaseOperator], values: Sequence[float]) -> List[complex]:
        state = Statevector.from_instruction(self._bind(values))
        return [state.expectation_value(obs) for obs in observables]

    def _evaluate_shots(self, observables: Iterable[BaseOperator], values: Sequence[float], shots: int) -> List[complex]:
        bound_circ = self._bind(values)
        bound_circ = transpile(bound_circ, optimization_level=1)
        backend = AerSimulator(shots=shots)
        results: List[complex] = []
        for obs in observables:
            pauli = PauliSumOp.from_operator(obs)
            expectation = PauliExpectation().convert(pauli)
            expr = StateFn(bound_circ, is_measurement=True) @ expectation
            sampler = CircuitSampler(backend)
            measured = sampler.convert(expr)
            value = measured.eval().real
            results.append(value)
        return results

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            if shots is None:
                row = self._evaluate_exact(observables, values)
            else:
                row = self._evaluate_shots(observables, values, shots)
            results.append(row)
        return results

    def predict(self, parameter_sets: Sequence[Sequence[float]]) -> np.ndarray:
        """Return statevector amplitudes for each parameter set."""
        amplitudes: List[np.ndarray] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            amplitudes.append(state.data)
        return np.array(amplitudes)

    def _add_noise(
        self,
        data: List[List[complex]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Add Gaussian shot noise to the deterministic estimator output."""
        if shots is None:
            return data
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in data:
            noisy_row = [
                complex(rng.normal(val.real, max(1e-6, 1 / shots)),
                        rng.normal(val.imag, max(1e-6, 1 / shots)))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

    def evaluate_with_noise(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Evaluate with optional shot noise."""
        raw = self.evaluate(observables, parameter_sets, shots=shots)
        return self._add_noise(raw, shots, seed)


__all__ = ["FastHybridEstimator"]
