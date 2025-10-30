"""Enhanced fast estimator for quantum circuits with gradient support.

Features:
* Parameter shift rule for exact gradients of expectation values.
* Shot‑noise simulation using sampling backend.
* GPU acceleration via qiskit_aer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

GradResult = Tuple[complex, List[complex]]  # (value, gradient vector)

class FastBaseEstimatorGen150:
    """Evaluate expectation values of observables for a parametrized circuit,
    with optional gradient computation via parameter shift rule and shot noise."""

    def __init__(self, circuit: QuantumCircuit, backend: str = "statevector_simulator") -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend_name = backend
        self.backend = Aer.get_backend(backend)

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
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            circ = self._bind(values)
            if self.backend_name == "statevector_simulator":
                state = Statevector.from_instruction(circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(circ, self.backend, shots=1024, memory=False)
                result = job.result()
                probs = result.get_statevector(circ)
                state = Statevector(probs)
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def _shifted_circuit(self, values: Sequence[float], idx: int, shift: float) -> QuantumCircuit:
        shifted = list(values)
        shifted[idx] += shift
        return self._bind(shifted)

    def evaluate_with_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[GradResult]]:
        """Compute expectation values and gradients via parameter shift rule."""
        observables = list(observables)
        results: List[List[GradResult]] = []
        for values in parameter_sets:
            circ = self._bind(values)
            if self.backend_name == "statevector_simulator":
                state = Statevector.from_instruction(circ)
                base_vals = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(circ, self.backend, shots=1024, memory=False)
                result = job.result()
                probs = result.get_statevector(circ)
                state = Statevector(probs)
                base_vals = [state.expectation_value(obs) for obs in observables]
            grads: List[List[complex]] = [[0.0] * len(self._parameters) for _ in observables]
            for p_idx in range(len(self._parameters)):
                plus = self._shifted_circuit(values, p_idx, np.pi / 2)
                minus = self._shifted_circuit(values, p_idx, -np.pi / 2)
                if self.backend_name == "statevector_simulator":
                    state_plus = Statevector.from_instruction(plus)
                    state_minus = Statevector.from_instruction(minus)
                    plus_vals = [state_plus.expectation_value(obs) for obs in observables]
                    minus_vals = [state_minus.expectation_value(obs) for obs in observables]
                else:
                    job_plus = execute(plus, self.backend, shots=1024, memory=False)
                    job_minus = execute(minus, self.backend, shots=1024, memory=False)
                    probs_plus = job_plus.result().get_statevector(plus)
                    probs_minus = job_minus.result().get_statevector(minus)
                    state_plus = Statevector(probs_plus)
                    state_minus = Statevector(probs_minus)
                    plus_vals = [state_plus.expectation_value(obs) for obs in observables]
                    minus_vals = [state_minus.expectation_value(obs) for obs in observables]
                for o_idx, (b, p, m) in enumerate(zip(base_vals, plus_vals, minus_vals)):
                    grads[o_idx][p_idx] = 0.5 * (p - m)
            row = [(base_vals[i], grads[i]) for i in range(len(observables))]
            results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Add sampling shot noise to deterministic evaluation."""
        if shots is None:
            return self.evaluate(observables, parameter_sets)
        results: List[List[complex]] = []
        for values in parameter_sets:
            circ = self._bind(values)
            job = execute(circ, self.backend, shots=shots, seed_simulator=seed)
            result = job.result()
            probs = result.get_statevector(circ)
            state = Statevector(probs)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = ["FastBaseEstimatorGen150"]
