"""
HybridEstimator: Quantum-only estimator that evaluates a Qiskit circuit
with optional shot‑noise simulation.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, Sequence, List, Optional

class QuantumCircuitEstimator:
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

class HybridEstimator:
    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._estimator = QuantumCircuitEstimator(circuit)
        self.shots = shots
        self.seed = seed

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        raw = self._estimator.evaluate(observables, parameter_sets)
        if self.shots is None:
            return [[float(v.real) for v in row] for row in raw]
        rng = np.random.default_rng(self.seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy.append([float(rng.normal(v.real, max(1e-6, 1 / self.shots))) for v in row])
        return noisy

# Quantum transformer utilities -----------------------------------------------
def _quantum_attention(circuit: QuantumCircuit, qubits: Sequence[int], params: Sequence[float]) -> QuantumCircuit:
    for q, p in zip(qubits, params):
        circuit.ry(p, q)
    for i in range(len(qubits) - 1):
        circuit.cx(qubits[i], qubits[i + 1])
    return circuit

def _quantum_feedforward(circuit: QuantumCircuit, qubits: Sequence[int], params: Sequence[float]) -> QuantumCircuit:
    for q, p in zip(qubits, params):
        circuit.rz(p, q)
    circuit.barrier()
    return circuit

def build_quantum_transformer(num_qubits: int, depth: int, parameter_count: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for layer in range(depth):
        attn_params = [float(layer * 0.1 + i) for i in range(parameter_count)]
        qc = _quantum_attention(qc, list(range(num_qubits)), attn_params)
        ffn_params = [float(layer * 0.2 + i) for i in range(parameter_count)]
        qc = _quantum_feedforward(qc, list(range(num_qubits)), ffn_params)
    return qc

__all__ = [
    "HybridEstimator",
    "build_quantum_transformer",
]
