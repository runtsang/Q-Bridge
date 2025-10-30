"""Extended quantum estimator that supports shot‑noise simulation and analytic gradients via the parameter‑shift rule."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimatorExtended:
    """Quantum estimator that evaluates expectation values, supports shot noise, and computes gradients.

    Features
    ----------
    * Parameter‑shift rule for analytic gradients.
    * Optional shot‑noise simulation by sampling measurement outcomes.
    * Flexible handling of multiple observables per circuit.
    """

    def __init__(self, circuit: QuantumCircuit, backend=None):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend or Aer.get_backend("qasm_simulator")

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

    def evaluate_with_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Estimate expectation values by sampling measurement outcomes.

        Parameters
        ----------
        shots : int
            Number of measurement shots per evaluation.
        seed : int | None, optional
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        results: List[List[complex]] = []
        for values in parameter_sets:
            circ = self._bind(values)
            # Add measurements in computational basis for each qubit
            circ.measure_all()
            job = execute(circ, backend=self.backend, shots=shots)
            counts = job.result().get_counts()
            row: List[complex] = []
            for obs in observables:
                # For simplicity, we assume obs is a PauliSumOp or similar.
                # Convert counts to expectation value assuming Z‑basis measurement.
                exp = self._counts_to_expectation(counts, obs)
                row.append(exp)
            results.append(row)
        return results

    def _counts_to_expectation(self, counts: dict[str, int], observable: BaseOperator) -> complex:
        """Convert measurement counts to expectation value for a Pauli observable."""
        # This is a simplified implementation that works for Z‑type observables.
        shots = sum(counts.values())
        pos = sum(v for k, v in counts.items() if k.count("1") % 2 == 0)
        neg = shots - pos
        return (pos - neg) / shots

    def gradient(
        self,
        observable: BaseOperator,
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute analytic gradients via the parameter‑shift rule."""
        results: List[List[complex]] = []
        for values in parameter_sets:
            grads: List[complex] = []
            for idx in range(len(self._parameters)):
                shift = np.pi / 2
                plus = list(values)
                minus = list(values)
                plus[idx] += shift
                minus[idx] -= shift
                exp_plus = self.evaluate([observable], [plus])[0][0]
                exp_minus = self.evaluate([observable], [minus])[0][0]
                grads.append((exp_plus - exp_minus) / (2 * np.sin(shift)))
            results.append(grads)
        return results

    def evaluate_and_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[tuple[List[complex], List[List[complex]]]]:
        """Return expectation values and gradients for each observable."""
        results: List[tuple[List[complex], List[List[complex]]]] = []
        for values in parameter_sets:
            evs = self.evaluate(observables, [values])[0]
            grads_per_obs: List[List[complex]] = []
            for obs in observables:
                grads_per_obs.append(self.gradient(obs, [values])[0])
            results.append((evs, grads_per_obs))
        return results


__all__ = ["FastBaseEstimatorExtended"]
