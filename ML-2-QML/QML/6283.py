"""Enhanced fast estimator for parametrised quantum circuits.

Features added:
* Parameter‑shift gradient estimation.
* Shot‑noise simulation with configurable shot number.
* Support for multiple observables per circuit.
* Automatic parameter binding with type safety.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit.

    Parameters
    ----------
    circuit: QuantumCircuit
        A circuit containing symbolic parameters.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters: List[str] = [str(p) for p in circuit.parameters]
        self._param_len = len(self._parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= self._param_len:
            raise ValueError(f"Parameter count mismatch: expected {self._param_len}, got {len(parameter_values)}")
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

    def evaluate_with_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Return noisy expectation values using a classical simulator with shots."""
        rng = np.random.default_rng(seed)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            probs = state.probabilities()
            # sample measurement outcomes
            outcomes = rng.choice(len(probs), size=shots, p=probs)
            # compute expectation from sampled outcomes
            row: List[complex] = []
            for obs in observables:
                expect = sum(obs.data[0, 0] * (outcomes == 0) +
                              obs.data[1, 1] * (outcomes == 1)) / shots
                row.append(expect)
            results.append(row)
        return results

    def parameter_shift_gradient(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shift: float = np.pi / 2,
    ) -> List[List[torch.Tensor]]:
        """Estimate gradients of observables w.r.t. circuit parameters using the
        parameter‑shift rule.

        Returns a list of gradient tensors for each observable and parameter set.
        """
        import torch
        observables = list(observables)
        gradients: List[List[torch.Tensor]] = []
        for values in parameter_sets:
            grads_per_obs: List[torch.Tensor] = []
            for obs in observables:
                grad: List[float] = []
                for i in range(self._param_len):
                    plus = list(values)
                    minus = list(values)
                    plus[i] += shift
                    minus[i] -= shift
                    exp_plus = Statevector.from_instruction(self._bind(plus)).expectation_value(obs)
                    exp_minus = Statevector.from_instruction(self._bind(minus)).expectation_value(obs)
                    grad.append(float((exp_plus - exp_minus) / (2 * np.sin(shift))))
                grads_per_obs.append(torch.tensor(grad, dtype=torch.float32))
            gradients.append(grads_per_obs)
        return gradients

__all__ = ["FastBaseEstimator"]
