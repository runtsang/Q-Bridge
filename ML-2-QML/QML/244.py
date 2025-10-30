"""Enhanced quantum estimator for parameterised circuits.

Features added:
- Shot‑noise simulation via Aer QASM simulator.
- Parameter‑shift gradient computation for arbitrary observables.
- Batch evaluation with vectorised parameter sets.
- Flexible backend selection and shot control.
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
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[object] = None,
        shots: int | None = 1024,
        seed: Optional[int] = None,
    ) -> None:
        self.circuit = circuit
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.seed = seed
        self.parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------
    def evaluate_batch(
        self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]
    ) -> List[List[complex]]:
        """Vectorised evaluation that optionally uses the QASM simulator for shot noise."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            if self.shots:
                job = execute(
                    self._bind(values),
                    backend=self.backend,
                    shots=self.shots,
                    seed_simulator=self.seed,
                )
                result = job.result()
                counts = result.get_counts()
                sv = Statevector.from_counts(counts)
                row = [sv.expectation_value(obs) for obs in observables]
            else:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    # ------------------------------------------------------------------
    # Parameter‑shift gradient
    # ------------------------------------------------------------------
    def _parameter_shift(
        self, observable: BaseOperator, params: Sequence[float]
    ) -> complex:
        shift = np.pi / 2
        grad = 0.0
        for i, p in enumerate(params):
            p_plus = list(params)
            p_minus = list(params)
            p_plus[i] += shift
            p_minus[i] -= shift
            e_plus = self._expectation(observable, p_plus)
            e_minus = self._expectation(observable, p_minus)
            grad += (e_plus - e_minus) / 2
        return grad

    def _expectation(self, observable: BaseOperator, params: Sequence[float]) -> complex:
        bound_circ = self._bind(params)
        if self.shots:
            job = execute(
                bound_circ,
                backend=self.backend,
                shots=self.shots,
                seed_simulator=self.seed,
            )
            result = job.result()
            counts = result.get_counts()
            sv = Statevector.from_counts(counts)
            return sv.expectation_value(observable)
        else:
            state = Statevector.from_instruction(bound_circ)
            return state.expectation_value(observable)

    def evaluate_gradients(
        self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]
    ) -> List[List[complex]]:
        """Return gradients of each observable w.r.t. the circuit parameters."""
        observables = list(observables)
        grads: List[List[complex]] = []
        for params in parameter_sets:
            grad_row = [self._parameter_shift(obs, params) for obs in observables]
            grads.append(grad_row)
        return grads

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def set_shots(self, shots: int | None) -> None:
        """Set the number of shots for stochastic evaluation."""
        self.shots = shots

    def set_seed(self, seed: int | None) -> None:
        """Set the simulator seed for reproducibility."""
        self.seed = seed


__all__ = ["FastBaseEstimator"]
