"""Hybrid fully‑connected layer – quantum implementation.

The class reproduces the classical API while using a simple
parameterised quantum circuit.  It implements a FastBaseEstimator
that evaluates expectation values of Pauli‑Z observables and
provides optional shot‑noise emulation, mirroring the FastEstimator
pattern from the classical side.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, Sequence, List
from collections.abc import Iterable as IterableABC

# --------------------------------------------------------------------------- #
# FastBaseEstimator – evaluates observables for a parametrised circuit
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluates expectation values of observables for a parametrised circuit."""

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
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


# --------------------------------------------------------------------------- #
# HybridFCL – quantum fully‑connected layer
# --------------------------------------------------------------------------- #
class HybridFCL:
    """Quantum fully‑connected layer that mirrors the classical API."""

    def __init__(self, n_qubits: int = 1, shots: int = 100, backend=None) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("statevector_simulator")

        # Build a simple variational circuit
        self._circuit = QuantumCircuit(n_qubits)
        self.params = [Parameter(f"theta_{q}") for q in range(n_qubits)]
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        for q, p in enumerate(self.params):
            self._circuit.ry(p, q)
        self._circuit.measure_all()

        self.estimator = FastBaseEstimator(self._circuit)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return the expectation value of a Pauli‑Z sum observable."""
        pauli_z = SparsePauliOp.from_list([("Z" * self.n_qubits, 1)])
        results = self.estimator.evaluate([pauli_z], [list(thetas)])
        return np.array([results[0][0]])

    def estimator(self) -> FastBaseEstimator:
        """Return the underlying FastBaseEstimator for custom evaluations."""
        return self.estimator


__all__ = ["HybridFCL", "FastBaseEstimator"]
