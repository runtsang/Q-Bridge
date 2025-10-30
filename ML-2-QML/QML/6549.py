from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class FastEstimator:
    """Evaluate expectation values of observables for a parametrized circuit with optional shot noise and gradients.

    Parameters
    ----------
    circuit : QuantumCircuit
        The parametrized quantum circuit.
    shots : int | None, optional
        Number of shots for measurement simulation. If None, exact expectation values are returned.
    """

    def __init__(self, circuit: QuantumCircuit, shots: int | None = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._shots = shots

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
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_and_grad(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shift: float = np.pi / 2,
    ) -> tuple[List[List[complex]], List[List[List[float]]]]:
        """Compute expectation values and gradients using the parameter‑shift rule.

        Parameters
        ----------
        shift : float, optional
            Shift value for the parameter‑shift rule (default π/2).

        Returns
        -------
        tuple
            (values, gradients) where gradients are lists of lists of floats matching the
            observable‑by‑parameter dimensions.
        """
        observables = list(observables)
        values: List[List[complex]] = []
        grads: List[List[List[float]]] = []

        for params in parameter_sets:
            base_vals = self.evaluate(observables, [params])[0]
            grad_obs: List[List[float]] = []
            for obs_idx, _ in enumerate(observables):
                grad_params: List[float] = []
                for idx, _ in enumerate(self._parameters):
                    shifted_plus = list(params)
                    shifted_minus = list(params)
                    shifted_plus[idx] += shift
                    shifted_minus[idx] -= shift
                    plus_vals = self.evaluate(observables, [shifted_plus])[0]
                    minus_vals = self.evaluate(observables, [shifted_minus])[0]
                    grad = (plus_vals[obs_idx] - minus_vals[obs_idx]) * 0.5
                    grad_params.append(float(grad))
                grad_obs.append(grad_params)
            values.append(base_vals)
            grads.append(grad_obs)
        return values, grads

    def evaluate_noisy(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Add Gaussian shot noise to deterministic expectation values.

        Parameters
        ----------
        shots : int | None
            Number of measurement shots; if None, no noise added.
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        List[List[complex]]
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean.real, max(1e-6, 1 / np.sqrt(shots)))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastEstimator"]
