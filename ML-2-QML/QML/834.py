"""Enhanced fast quantum estimator for parameterized circuits.

Features:
- Batch evaluation of expectation values for multiple observables.
- Shot‑noise simulation via Gaussian perturbations.
- Parameter‑shift gradient calculation.
- Flexible backend selection (local state‑vector or real devices).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector, Operator
from qiskit.opflow import PauliSumOp

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[Backend] = None,
        shots: Optional[int] = None,
    ) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend = backend or AerSimulator(method="statevector")
        self.shots = shots

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def _expectation(
        self,
        observable: Operator,
        circuit: QuantumCircuit,
    ) -> complex:
        if self.shots is None:
            # State‑vector evaluation
            state = Statevector.from_instruction(circuit)
            return state.expectation_value(observable).real
        else:
            # Backend execution with shots
            job = execute(
                circuit,
                backend=self.backend,
                shots=self.shots,
                memory=False,
            )
            result = job.result()
            counts = result.get_counts()
            exp = 0.0
            for bitstring, count in counts.items():
                parity = (-1) ** bitstring.count("1")
                exp += parity * count
            return exp / self.shots

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return a 2‑D list of expectation values."""
        obs_list = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            circ = self._bind(values)
            row = [self._expectation(obs, circ) for obs in obs_list]
            results.append(row)
        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Same as :meth:`evaluate` but adds Gaussian shot noise."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = [[float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row] for row in raw]
        return noisy

    def gradient(
        self,
        observable: Operator,
        parameter_sets: Sequence[Sequence[float]],
        *,
        step: float = 1e-3,
    ) -> List[List[float]]:
        """Parameter‑shift gradient of a single observable.

        Returns a list of shape (num_sets, num_params).
        """
        grad_list: List[List[float]] = []
        for values in parameter_sets:
            grad = []
            for i, _ in enumerate(self.parameters):
                circ_plus = self._bind([v + (step if idx == i else 0) for idx, v in enumerate(values)])
                circ_minus = self._bind([v - (step if idx == i else 0) for idx, v in enumerate(values)])
                exp_plus = self._expectation(observable, circ_plus)
                exp_minus = self._expectation(observable, circ_minus)
                grad.append((exp_plus - exp_minus) / (2 * step))
            grad_list.append(grad)
        return grad_list

__all__ = ["FastBaseEstimator"]
