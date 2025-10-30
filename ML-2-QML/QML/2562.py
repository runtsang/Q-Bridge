"""Hybrid estimator for parameterised quantum circuits with support for shot noise.

This module defines ``HybridEstimator`` for evaluating a quantum circuit against
a set of observables.  It mirrors the classical estimator’s interface, enabling
direct substitution in hybrid workflows.  The ``FCL`` helper builds a simple
parameterised circuit that emulates a fully‑connected layer using a single qubit.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List

class HybridEstimator:
    """Evaluate a parameterised quantum circuit for a list of observables."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a new circuit with parameters bound to the supplied values."""
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return a matrix of expectation values.

        Parameters
        ----------
        observables:
            Qiskit operators whose expectation values are to be computed.
        parameter_sets:
            Sequence of parameter vectors to bind to the circuit.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results


class FastEstimator(HybridEstimator):
    """Add optional shot‑noise to the quantum estimator by running the circuit."""

    def __init__(self, circuit: QuantumCircuit, shots: int = 1024) -> None:
        super().__init__(circuit)
        self.shots = shots

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Execute the circuit on a simulator and return noisy expectation values."""
        from qiskit import Aer, execute

        backend = Aer.get_backend("qasm_simulator")
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            job = execute(bound, backend, shots=self.shots)
            result = job.result()
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            # add shot‑noise by sampling the expectation value
            noisy_row = [
                complex(np.random.normal(float(val.real), 1 / np.sqrt(self.shots)))
                for val in row
            ]
            results.append(noisy_row)
        return results


def FCL() -> QuantumCircuit:
    """Return a simple parameterised circuit that mimics a fully‑connected layer.

    The circuit applies a Hadamard gate, a parameterised Ry rotation, and
    measures in the computational basis.  The expectation value of the
    measurement outcome is returned as a single‑value array.
    """
    class ParamCircuit:
        def __init__(self, n_qubits: int = 1, shots: int = 100) -> None:
            self._circuit = QuantumCircuit(n_qubits)
            self.theta = QuantumCircuit.parameter("theta")
            self._circuit.h(range(n_qubits))
            self._circuit.ry(self.theta, range(n_qubits))
            self._circuit.measure_all()
            self.shots = shots

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            from qiskit import Aer, execute

            backend = Aer.get_backend("qasm_simulator")
            job = execute(
                self._circuit,
                backend,
                shots=self.shots,
                parameter_binds=[{self.theta: theta} for theta in thetas],
            )
            result = job.result().get_counts(self._circuit)
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            probabilities = counts / self.shots
            expectation = np.sum(states * probabilities)
            return np.array([expectation])

    return ParamCircuit()


__all__ = ["HybridEstimator", "FastEstimator", "FCL"]
