"""Quantum estimator with flexible simulation back‑ends.

Features
--------
* Supports both state‑vector and shot‑based simulation via Qiskit Aer.
* Optional noise models (e.g., amplitude damping, depolarizing).
* Caches bound circuits to avoid re‑binding overhead.
* Allows evaluation of multiple observables per parameter set.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimatorGen256:
    """Evaluate expectation values of observables for a parametrized circuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: str = "statevector",
        shots: Optional[int] = None,
        noise_model: Optional[object] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._backend_name = backend
        self._shots = shots
        self._noise_model = noise_model
        self._backend = self._select_backend()
        self._cache: dict[tuple[float,...], QuantumCircuit] = {}

    def _select_backend(self):
        if self._backend_name == "statevector":
            return Aer.get_backend("statevector_simulator")
        elif self._backend_name == "qasm":
            backend = Aer.get_backend("qasm_simulator")
            if self._noise_model:
                backend = Aer.get_backend("qasm_simulator", noise_model=self._noise_model)
            return backend
        else:
            raise ValueError(f"Unsupported backend: {self._backend_name}")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        key = tuple(parameter_values)
        if key in self._cache:
            return self._cache[key]
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        bound = self._circuit.assign_parameters(mapping, inplace=False)
        self._cache[key] = bound
        return bound

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound_circ = self._bind(values)

            if self._backend_name == "statevector":
                state = Statevector.from_instruction(bound_circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:  # shot‑based simulation
                job = execute(
                    bound_circ,
                    backend=self._backend,
                    shots=self._shots or 1024,
                    noise_model=self._noise_model,
                )
                result = job.result()
                moments = result.get_memory()
                # Compute expectation from measurement counts
                row = []
                for obs in observables:
                    # Very naive expectation: average over bitstrings
                    exp = 0.0
                    for bitstr, count in result.get_counts(bound_circ).items():
                        exp += count * self._bitstring_expectation(bitstr, obs)
                    exp /= sum(result.get_counts(bound_circ).values())
                    row.append(complex(exp))
            results.append(row)
        return results

    @staticmethod
    def _bitstring_expectation(bitstr: str, observable: BaseOperator) -> float:
        """Map a bitstring to an eigenvalue of the observable."""
        # Placeholder: assumes PauliZ tensor product; extend as needed.
        eigenvalue = 1.0
        for qubit, bit in enumerate(reversed(bitstr)):
            if isinstance(observable, BaseOperator) and observable.data.shape[0] > 1:
                # Simplistic PauliZ eigenvalue mapping
                if bit == "1":
                    eigenvalue *= -1
        return eigenvalue


__all__ = ["FastBaseEstimatorGen256"]
