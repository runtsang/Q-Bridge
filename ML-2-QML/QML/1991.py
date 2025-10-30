"""Hybrid estimator for quantum circuits with shot‑noise simulation and parameter‑shift gradients."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values for a parameterised quantum circuit with optional shot noise
    and provide gradients via the parameter‑shift rule."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable,
        optionally adding shot‑noise via a classical Monte‑Carlo simulation."""
        observables = list(observables)
        results: List[List[complex]] = []
        rng = np.random.default_rng(seed)
        for params in parameter_sets:
            state = Statevector.from_instruction(self._bind(params))
            row = [state.expectation_value(obs) for obs in observables]
            if shots is not None:
                probs = state.probabilities()
                samples = rng.choice(len(probs), size=shots, p=probs)
                noisy_row = []
                for obs in observables:
                    # Assume Pauli observables with eigenvalues ±1 for simplicity
                    if hasattr(obs, "eigenvalues"):
                        eigs = obs.eigenvalues()
                        eig_vals = [eigs[i] for i in samples]
                        noisy_row.append(np.mean(eig_vals))
                    else:
                        noisy_row.append(row[observables.index(obs)])
                row = noisy_row
            results.append(row)
        return results

    def gradient(
        self,
        observable: BaseOperator,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shift: float = np.pi / 2,
    ) -> List[List[float]]:
        """Return gradients of the expectation value w.r.t. each circuit parameter
        using the parameter‑shift rule."""
        grads: List[List[float]] = []
        for params in parameter_sets:
            grad_params: List[float] = []
            for i, _ in enumerate(self._params):
                shift_plus = list(params)
                shift_minus = list(params)
                shift_plus[i] += shift
                shift_minus[i] -= shift
                exp_plus = Statevector.from_instruction(
                    self._bind(shift_plus)
                ).expectation_value(observable)
                exp_minus = Statevector.from_instruction(
                    self._bind(shift_minus)
                ).expectation_value(observable)
                grad = 0.5 * (exp_plus - exp_minus)
                grad_params.append(float(grad))
            grads.append(grad_params)
        return grads


__all__ = ["FastBaseEstimator"]
