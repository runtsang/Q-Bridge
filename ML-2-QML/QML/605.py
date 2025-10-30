"""Hybrid estimator for quantum circuits with shot noise support."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional, Union

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastHybridEstimator:
    """Evaluate a Qiskit quantum circuit on batches of parameters with optional shot noise.

    Parameters
    ----------
    circuit
        A Qiskit QuantumCircuit with symbolic parameters.
    shots
        Number of shots; if None, exact expectation values are returned.
    seed
        Random seed for the simulator.
    """
    def __init__(self, circuit: QuantumCircuit, shots: Optional[int] = None, seed: Optional[int] = None) -> None:
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("circuit must be a qiskit.circuit.QuantumCircuit")
        self.circuit = circuit
        self.shots = shots
        self.seed = seed

    def _bind(self, params: Sequence[float]) -> QuantumCircuit:
        if len(params)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, params))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            if self.shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                backend = Aer.get_backend("qasm_simulator")
                job = execute(bound, backend=backend, shots=self.shots, seed_simulator=self.seed)
                result = job.result()
                counts = result.get_counts(bound)
                # Convert counts to expectation values
                row = []
                for obs in observables:
                    exp = 0.0
                    for bitstring, count in counts.items():
                        exp += self._bitstring_expectation(bitstring, obs) * count
                    exp /= self.shots
                    row.append(exp)
            results.append(row)
        return results

    @staticmethod
    def _bitstring_expectation(bitstring: str, observable: BaseOperator) -> float:
        """Compute expectation contribution from a single measurement outcome.
        Assumes observable is diagonal in computational basis (e.g., Pauli Z operators)."""
        # Simplistic: support only tensor products of Pauli Z
        expectation = 1.0
        for qubit, bit in enumerate(reversed(bitstring)):
            if observable.coeffs[0] == 1:  # placeholder for Pauli Z on this qubit
                expectation *= 1 if bit == "0" else -1
        return expectation


__all__ = ["FastHybridEstimator"]
