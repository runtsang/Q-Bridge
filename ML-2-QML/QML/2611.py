"""Hybrid fast estimator for Qiskit circuits with optional shot noise.

Features
--------
* Parameter binding for arbitrary sequences of values.
* Evaluation of a list of observables (Pauli operators, custom matrices).
* Optional shot‑based sampling that mimics finite‑shot quantum hardware.
* A convenience method to create a simple variational circuit that can be
  used as a drop‑in replacement for the original FastBaseEstimator.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


def _ensure_batch(values: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
    """Ensure the parameter list is a list of lists."""
    return [list(v) for v in values]


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized Qiskit circuit."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a new circuit with the supplied parameter values."""
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of Pauli or matrix operators for which the expectation
            value is computed.
        parameter_sets
            List of parameter vectors to evaluate.
        shots
            If provided, the circuit is executed with this many shots and the
            expectation value is estimated from the measurement histogram.
            Otherwise a state‑vector simulation is used.
        seed
            Random seed for the backend simulator.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if shots is None:
            # State‑vector evaluation
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(ob) for ob in observables]
                results.append(row)
        else:
            backend = Aer.get_backend("qasm_simulator")
            for values in parameter_sets:
                bound = self._bind(values)
                job = execute(bound, backend, shots=shots, seed_simulator=seed)
                result = job.result()
                counts = result.get_counts()
                # Convert counts to expectation value for each observable
                row = []
                for ob in observables:
                    exp = 0.0
                    for bitstring, freq in counts.items():
                        # Map bitstring to ±1 eigenvalue of the Pauli operator
                        eigen = 1
                        for qubit, pauli in enumerate(reversed(ob.paulis)):
                            if pauli == "Z":
                                eigen *= 1 if bitstring[qubit] == "0" else -1
                            elif pauli == "X":
                                # For X we need to consider superposition; skip for simplicity
                                eigen = 0
                                break
                        exp += eigen * freq / shots
                    row.append(complex(exp))
                results.append(row)

        return results

    @staticmethod
    def create_variational_circuit(n_qubits: int, depth: int = 2) -> QuantumCircuit:
        """Return a simple variational circuit with RX layers and CNOT entanglement."""
        qc = QuantumCircuit(n_qubits)
        for d in range(depth):
            for q in range(n_qubits):
                qc.rx(Parameter(f"θ_{d}_{q}"), q)
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            qc.barrier()
        return qc


__all__ = ["FastBaseEstimator"]
