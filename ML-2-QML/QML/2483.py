"""Hybrid estimator that evaluates a parameterised quantum circuit on a
backend and aggregates expectation values of observables.  The class
mirrors the API of the classical estimator so that experiments can be
run on either side with identical code.  The implementation uses Qiskit
to build circuits, but the design is agnostic to the backend – any
Aer or real device can be supplied.  A quantum self‑attention block is
provided that reproduces the structure of the classical version.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List, Optional

class QuantumSelfAttention:
    """
    Quantum self‑attention block that encodes a small attention‑style
    circuit.  Parameters are supplied externally to keep the class
    stateless, matching the interface of ClassicalSelfAttention.
    """

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


class FastBaseEstimator:
    """
    Quantum estimator that evaluates a parameterised circuit on a backend
    and returns expectation values of supplied observables.
    """

    def __init__(self, circuit: QuantumCircuit):
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
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        backend: Optional[object] = None,
    ) -> List[List[complex]]:
        if shots is None:
            return self.evaluate(observables, parameter_sets)
        if backend is None:
            backend = Aer.get_backend("qasm_simulator")
        results: List[List[complex]] = []
        for values in parameter_sets:
            circuit = self._bind(values)
            job = execute(circuit, backend, shots=shots)
            counts = job.result().get_counts(circuit)
            exp = 0.0
            for bitstring, freq in counts.items():
                bit = int(bitstring, 2)
                parity = (-1) ** bit
                exp += parity * freq / shots
            results.append([exp])
        return results


__all__ = ["FastBaseEstimator", "QuantumSelfAttention"]
