"""Hybrid estimator evaluating quantum circuits with optional shot noise and multiple observables."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized quantum circuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[qiskit.providers.Backend] = None,
        shots: Optional[int] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Optional[Iterable[BaseOperator]] = None,
        parameter_sets: Sequence[Sequence[float]] = (),
    ) -> List[List[complex]]:
        """
        Parameters
        ----------
        observables : Optional[Iterable[BaseOperator]]
            A list of operators.  If ``None`` the identity is used.
        parameter_sets : Sequence[Sequence[float]]
            A list of parameter vectors that will be fed to the circuit.
        """
        observables = list(observables or [qiskit.quantum_info.Operator(np.eye(2 ** self._circuit.num_qubits))])
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound_qc = self._bind(values)

            if self.shots is None:
                # Deterministic Statevector evaluation
                state = Statevector.from_instruction(bound_qc)
                row = [state.expectation_value(op) for op in observables]
            else:
                # Shot‑noise simulation using Aer
                job = qiskit.execute(
                    bound_qc,
                    self.backend,
                    shots=self.shots,
                )
                result = job.result()
                counts = result.get_counts(bound_qc)
                probs = {k: v / self.shots for k, v in counts.items()}
                # Build a probability distribution over basis states
                expectation = sum(
                    complex(int(state, 2)) * prob for state, prob in probs.items()
                )
                row = [expectation] * len(observables)  # identical for all observables in this toy demo
            results.append(row)

        return results


def FCL() -> QuantumCircuit:
    """
    A simple parameterised quantum circuit for a fully connected layer.

    The circuit applies a Hadamard layer, a parameterised rotation Ry(theta)
    on each qubit, and measures all qubits.  The expectation is computed
    as the weighted average of observed bit‑strings.
    """
    class QuantumCircuitWrapper:
        def __init__(self, n_qubits: int = 1, shots: int = 100) -> None:
            self._qc = qiskit.QuantumCircuit(n_qubits)
            self.theta = qiskit.circuit.Parameter("theta")
            self._qc.h(range(n_qubits))
            self._qc.ry(self.theta, range(n_qubits))
            self._qc.measure_all()
            self.shots = shots
            self.backend = qiskit.Aer.get_backend("qasm_simulator")

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            job = qiskit.execute(
                self._qc,
                self.backend,
                shots=self.shots,
                parameter_binds=[{self.theta: theta} for theta in thetas],
            )
            result = job.result()
            counts = result.get_counts(self._qc)
            probs = {k: v / self.shots for k, v in counts.items()}
            expectation = sum(complex(int(state, 2)) * prob for state, prob in probs.items())
            return np.array([expectation])

    return QuantumCircuitWrapper()


__all__ = ["FastBaseEstimator", "FCL"]
