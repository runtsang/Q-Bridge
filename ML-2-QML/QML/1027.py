"""Advanced Fast Estimator for Qiskit circuits with shot simulation and gradient estimation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class AdvancedFastEstimator:
    """Estimator for parametrized quantum circuits with shot‑based simulation and parameter‑shift gradients."""
    def __init__(self, circuit: QuantumCircuit, backend=None) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend = backend or Aer.get_backend("statevector_simulator")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

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

    def evaluate_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Simulate the circuit with a finite number of shots, returning noisy expectation values."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            job = execute(
                bound,
                backend=self.backend,
                shots=shots,
                seed_simulator=seed,
                seed_transpiler=seed,
            )
            result = job.result()
            counts = result.get_counts()
            row = []
            for obs in observables:
                exp = 0.0
                for bitstring, count in counts.items():
                    idx = int(bitstring[::-1], 2)
                    exp += obs.data[idx, idx] * count
                exp /= shots
                row.append(complex(exp))
            results.append(row)
        return results

    def compute_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[complex]]]:
        """Return gradients of each observable w.r.t. each circuit parameter using the parameter‑shift rule."""
        observables = list(observables)
        gradients: List[List[List[complex]]] = []
        shift = np.pi / 2
        for params in parameter_sets:
            grads_per_obs: List[List[complex]] = []
            for obs in observables:
                grads_per_param: List[complex] = []
                for idx, _ in enumerate(self.parameters):
                    plus = list(params)
                    plus[idx] += shift
                    exp_plus = self.evaluate([obs], [plus])[0][0]
                    minus = list(params)
                    minus[idx] -= shift
                    exp_minus = self.evaluate([obs], [minus])[0][0]
                    grad = (exp_plus - exp_minus) / 2
                    grads_per_param.append(grad)
                grads_per_obs.append(grads_per_param)
            gradients.append(grads_per_obs)
        return gradients


__all__ = ["AdvancedFastEstimator"]
