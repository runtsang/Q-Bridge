"""Quantum estimator with parametrized self‑attention circuit and observables."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List

class QuantumSelfAttention:
    """Parametrized self‑attention style block built on Qiskit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build(self, rot: np.ndarray, ent: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rot[3 * i], i)
            qc.ry(rot[3 * i + 1], i)
            qc.rz(rot[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(ent[i], i, i + 1)
        qc.barrier()
        return qc

    def run(
        self,
        backend,
        rot: np.ndarray,
        ent: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        qc = self._build(rot, ent)
        job = execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for vals in parameter_sets:
            state = Statevector.from_instruction(self._bind(vals))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


__all__ = ["QuantumSelfAttention", "FastBaseEstimator"]
