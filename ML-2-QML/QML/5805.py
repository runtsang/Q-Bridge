"""Quantum estimator that evaluates expectation values for a parameterised circuit.

The implementation supports exact Statevector evaluation as well as
shot‑noise sampling via the Aer QASM simulator.  Observables are
expected to be qiskit.quantum_info.operators.BaseOperator instances.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values of a parameterised quantum circuit."""

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
        """Compute expectation values for each parameter set.

        Parameters
        ----------
        observables:
            Iterable of BaseOperator objects (e.g. PauliZ, X, etc.).
        parameter_sets:
            Iterable of parameter vectors for the circuit.
        shots:
            If provided, use the Aer QASM simulator to sample measurement
            outcomes and approximate the expectation value.
        seed:
            Random seed for the simulator.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if shots is None:
            # Exact Statevector evaluation
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
            return results

        # Shot‑noise simulation
        backend = Aer.get_backend("qasm_simulator")
        dim = 2 ** len(self._parameters)

        for values in parameter_sets:
            bound = self._bind(values)
            job = execute(bound, backend=backend, shots=shots, seed_simulator=seed)
            counts = job.result().get_counts(bound)
            probs = np.zeros(dim, dtype=np.complex128)
            for bitstring, count in counts.items():
                idx = int(bitstring[::-1], 2)  # Qiskit uses little‑endian
                probs[idx] = count / shots
            row: List[complex] = []
            for obs in observables:
                matrix = obs.to_matrix()
                diag = np.diag(matrix)
                exp = np.dot(probs, diag)
                row.append(complex(exp))
            results.append(row)
        return results


__all__ = ["FastBaseEstimator"]
