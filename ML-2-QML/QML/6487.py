"""Hybrid estimator implemented with Qiskit."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class HybridEstimator:
    """Evaluate a Qiskit circuit for a batch of parameter sets.

    Parameters
    ----------
    circuit : QuantumCircuit
        A parameterised circuit that implements the desired model.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def _apply_observables(
        self,
        state: Statevector,
        observables: Iterable[BaseOperator],
    ) -> List[complex]:
        return [state.expectation_value(op) for op in observables]

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """
        Run the circuit on each parameter set and compute all observables.

        When *shots* is provided a Gaussian shot‑noise model is added to
        the deterministic expectation values, mimicking a noisy quantum
        backend.
        """
        raw: List[List[complex]] = []
        for params in parameter_sets:
            circ = self._bind(params)
            state = Statevector.from_instruction(circ)
            raw.append(self._apply_observables(state, observables))

        if shots is None:
            return raw

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy.append(
                [rng.normal(mean, max(1e-6, 1 / shots)) for mean in row]
            )
        return noisy


__all__ = ["HybridEstimator"]
