"""Quantum self‑attention module for hybrid execution.

Provides the same SelfAttentionHybrid class but implements the quantum
circuit using Qiskit and the FastBaseEstimator adapted for quantum
expectation values.  Classical execution is handled by the ml module.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
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


class QuantumSelfAttention:
    """Quantum circuit representing a self‑attention style block."""
    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.circuit = QuantumCircuit(self.qr, self.cr)

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
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
    ):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


class SelfAttentionHybrid:
    """Hybrid self‑attention that can run classically (via the ml module)
    or quantumly (via this module)."""
    def __init__(self, embed_dim: int = 4, use_quantum: bool = True) -> None:
        self.embed_dim = embed_dim
        self.use_quantum = use_quantum
        if use_quantum:
            self.quantum = QuantumSelfAttention(embed_dim)
            self.estimator = FastBaseEstimator(self.quantum.circuit)
        else:
            raise RuntimeError("Classical execution handled by the ml module.")

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        backend=None,
        shots: int = 1024,
    ):
        if not self.use_quantum:
            raise RuntimeError("Quantum execution requires the qml module.")
        if backend is None:
            backend = Aer.get_backend("qasm_simulator")
        return self.quantum.run(backend, rotation_params, entangle_params, shots)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        return self.estimator.evaluate(observables, parameter_sets)


__all__ = ["SelfAttentionHybrid"]
