"""Quantum estimator with expectation, shot‑noise, and parameter‑shift gradients.

Supports a parametrized quantum circuit, optional noise model, shot‑count simulation,
and analytic gradients via the parameter‑shift rule.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.providers.aer import AerSimulator
from qiskit import execute
from qiskit.providers import Backend
from qiskit.providers.models import NoiseModel


class FastBaseEstimator:
    """Evaluate expectation values and gradients for a parametrized circuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        noise_model: Optional[NoiseModel] = None,
        shots: Optional[int] = None,
        backend: Optional[Backend] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.noise_model = noise_model
        self.shots = shots
        # If a backend is supplied, use it; otherwise fall back to AerSimulator.
        self.backend = backend or AerSimulator(noise_model=noise_model)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _expectation(self, circ: QuantumCircuit, observable: Operator) -> complex:
        """Compute expectation value of an observable on a circuit."""
        # Use statevector simulation for deterministic expectation.
        state = Statevector.from_instruction(circ)
        return state.expectation_value(observable)

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            circ = self._bind(values)
            if self.shots is None:
                # Deterministic state‑vector evaluation
                row = [self._expectation(circ, obs) for obs in observables]
            else:
                # Shot‑limited evaluation via AerSimulator
                job = execute(circ, backend=self.backend, shots=self.shots)
                result = job.result()
                # Retrieve the final statevector from the result
                state = Statevector.from_dict(result.get_statevector(circ))
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_gradients(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[np.ndarray]]:
        """Return parameter‑shift gradients of each observable w.r.t. circuit parameters."""
        observables = list(observables)
        grads_list: List[List[np.ndarray]] = []
        shift = np.pi / 2  # standard parameter‑shift value

        for values in parameter_sets:
            grad_row: List[np.ndarray] = []
            for observable in observables:
                grads: List[float] = []
                for i, _ in enumerate(self._parameters):
                    # Shift +π/2 and –π/2 for the i‑th parameter
                    plus_vals = list(values)
                    minus_vals = list(values)
                    plus_vals[i] += shift
                    minus_vals[i] -= shift
                    circ_plus = self._bind(plus_vals)
                    circ_minus = self._bind(minus_vals)
                    exp_plus = self._expectation(circ_plus, observable)
                    exp_minus = self._expectation(circ_minus, observable)
                    grad = 0.5 * (exp_plus - exp_minus)
                    grads.append(grad.real)  # gradient is real for Hermitian observables
                grad_row.append(np.array(grads))
            grads_list.append(grad_row)
        return grads_list


__all__ = ["FastBaseEstimator"]
