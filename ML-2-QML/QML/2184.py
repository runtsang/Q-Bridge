"""FastEstimator for quantum circuits using Qiskit."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastEstimator:
    """Fast estimator for parameterised quantum circuits.

    Supports expectation value evaluation, optional shot‑noise simulation
    through a simple sampling routine, and analytic gradients via the
    parameter‑shift rule.
    """

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
        """Compute expectation values for each parameter set.

        When ``shots`` is provided the backend performs a simple
        stochastic measurement by sampling from the state‑vector
        distribution; otherwise a noiseless state‑vector expectation is
        used.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))

            if shots is None:
                row = [state.expectation_value(obs) for obs in observables]
            else:
                probs = state.probabilities_dict()
                keys = list(probs.keys())
                weights = list(probs.values())
                rng = np.random.default_rng(seed)
                samples = rng.choice(keys, size=shots, p=weights)

                row = []
                for obs in observables:
                    # Evaluate observable on each sampled basis state
                    eigvals = np.array(
                        [
                            float(
                                obs.expectation_value(Statevector.from_label(bit))
                            )
                            for bit in samples
                        ]
                    )
                    row.append(eigvals.mean())
            results.append(row)
        return results

    def gradient(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[np.ndarray]]:
        """Analytic gradients via parameter‑shift rule.

        Returns a list of gradient matrices for each parameter set
        with shape ``(num_observables, num_params)``.
        """
        observables = list(observables)
        grads: List[List[np.ndarray]] = []

        for values in parameter_sets:
            grad_row: List[np.ndarray] = []
            for obs in observables:
                grad_vec = np.zeros(len(self._params))
                for i, _ in enumerate(self._params):
                    shift = np.pi / 2
                    pos = list(values)
                    neg = list(values)
                    pos[i] += shift
                    neg[i] -= shift
                    ep = Statevector.from_instruction(self._bind(pos)).expectation_value(obs)
                    em = Statevector.from_instruction(self._bind(neg)).expectation_value(obs)
                    grad_vec[i] = (ep - em) / 2.0
                grad_row.append(grad_vec)
            grads.append(grad_row)
        return grads

__all__ = ["FastEstimator"]
