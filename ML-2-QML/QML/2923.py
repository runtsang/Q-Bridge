"""Unified estimator for Qiskit quantum circuits with optional shot noise.

The estimator can compute exact expectation values via Statevector or simulate measurements
with a given number of shots, adding Gaussian noise to mimic shot noise.  The interface matches
the classical counterpart for ease of substitution."""
from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator
from collections.abc import Iterable, Sequence
from typing import List

class UnifiedEstimator:
    """Evaluate a Qiskit quantum circuit for batches of parameters and observables.

    The estimator can compute exact expectation values via Statevector or simulate measurements
    with a given number of shots, adding Gaussian noise to mimic shot noise.  The interface matches
    the classical counterpart for ease of substitution.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _evaluate_exact(self,
                        observables: Iterable[BaseOperator],
                        parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def _evaluate_shots(self,
                        observables: Iterable[BaseOperator],
                        parameter_sets: Sequence[Sequence[float]],
                        shots: int,
                        seed: int | None) -> List[List[complex]]:
        simulator = AerSimulator()
        results: List[List[complex]] = []
        for values in parameter_sets:
            circ = self._bind(values)
            # Extend circuit to measure all qubits in computational basis
            circ.measure_all()
            job = simulator.run(circ, shots=shots, seed_simulator=seed)
            result = job.result()
            counts = result.get_counts(circ)
            # Convert counts to statevector probabilities
            probs = {int(k, 2): v / shots for k, v in counts.items()}
            row = []
            for obs in observables:
                exp = 0.0 + 0.0j
                for bitstring, prob in probs.items():
                    # expectation value of PauliZ in computational basis
                    # We approximate by projecting onto eigenstates
                    exp += prob * obs.data[bitstring, bitstring]
                row.append(exp)
            results.append(row)
        return results

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[complex]]:
        """Compute expectation values, optionally using measurement shots."""
        if shots is None:
            return self._evaluate_exact(observables, parameter_sets)
        else:
            return self._evaluate_shots(observables, parameter_sets, shots, seed)


__all__ = ["UnifiedEstimator"]
