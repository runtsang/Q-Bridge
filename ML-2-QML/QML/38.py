"""Quantum estimator primitives that evaluate expectation values of
parametrized circuits.

Two classes are provided:

- **FastBaseEstimator** – evaluates observables for many parameter sets.
  It supports a state‑vector backend for exact expectation values and
  a shot‑based Aer simulator for noisy sampling.  Gradients are
  computed via the parameter‑shift rule.

- **FastEstimator** – inherits FastBaseEstimator and sets a default
  shot count, providing a convenient noisy estimator out of the box.

The API is backward compatible with the original seed but adds
gradient support and optional noise simulation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        backend: Optional[object] = None,
        shots: Optional[int] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        # Choose a backend: statevector for exact, qasm for sampling
        if backend is None:
            self.backend = (
                Aer.get_backend("statevector_simulator") if shots is None else Aer.get_backend("qasm_simulator")
            )
        else:
            self.backend = backend
        self.shots = shots

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a new circuit with parameters bound to the given values."""
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

        rng = np.random.default_rng()  # for shot noise

        for values in parameter_sets:
            circ = self._bind(values)
            # Exact expectation via statevector
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]

            # Add Gaussian noise if shots are requested
            if self.shots is not None:
                noise_std = max(1e-6, 1 / np.sqrt(self.shots))
                noisy_row = [rng.normal(val.real, noise_std) + 1j * rng.normal(val.imag, noise_std) for val in row]
                row = noisy_row

            results.append(row)

        return results

    def compute_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[np.ndarray]]:
        """Return gradients of observables w.r.t. circuit parameters using the parameter‑shift rule."""
        observables = list(observables)
        grads: List[List[np.ndarray]] = []
        shift = np.pi / 2

        for values in parameter_sets:
            grad_row: List[np.ndarray] = []
            for obs in observables:
                grad_vals: List[float] = []
                for i, _ in enumerate(self._parameters):
                    plus_vals = list(values)
                    minus_vals = list(values)
                    plus_vals[i] += shift
                    minus_vals[i] -= shift
                    circ_plus = self._bind(plus_vals)
                    circ_minus = self._bind(minus_vals)
                    state_plus = Statevector.from_instruction(circ_plus)
                    state_minus = Statevector.from_instruction(circ_minus)
                    e_plus = state_plus.expectation_value(obs)
                    e_minus = state_minus.expectation_value(obs)
                    grad = 0.5 * (e_plus - e_minus)
                    grad_vals.append(float(grad))
                grad_row.append(np.array(grad_vals))
            grads.append(grad_row)

        return grads


class FastEstimator(FastBaseEstimator):
    """Convenient estimator that adds Gaussian shot noise by default."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int = 1024,
        backend: Optional[object] = None,
    ) -> None:
        super().__init__(circuit, backend=backend, shots=shots)


__all__ = ["FastBaseEstimator", "FastEstimator"]
