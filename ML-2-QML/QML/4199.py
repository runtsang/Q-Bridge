"""Hybrid estimator combining Qiskit circuit evaluation and fraud‑detection photonic model."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit import Parameter

class FastHybridEstimator:
    """Evaluate expectation values of observables for a parametrized Qiskit circuit."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        if shots is None:
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(observable) for observable in observables]
                results.append(row)
        else:
            backend = Aer.get_backend("qasm_simulator")
            for values in parameter_sets:
                bound = self._bind(values)
                job = execute(bound, backend=backend, shots=shots, seed_simulator=seed)
                counts = job.result().get_counts()
                probs = {k: v / shots for k, v in counts.items()}
                row = []
                for obs in observables:
                    exp = sum(
                        (1 if "1" in state else -1) * prob
                        for state, prob in probs.items()
                    )
                    row.append(exp)
                results.append(row)
        return results


__all__ = ["FastHybridEstimator"]
