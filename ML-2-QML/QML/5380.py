"""Hybrid quantum‑classical self‑attention circuit.

The quantum part implements a parameterised Qiskit circuit that mimics the
classical self‑attention block: rotations encode the query/key/value
projections, controlled‑RX gates realise the entanglement, and a
graph‑based controlled‑Z mask is added to strengthen the attention
between highly correlated qubits.  Fraud‑detection style gates are
translated into qiskit primitives; the estimator follows the
`FastBaseEstimator` pattern and returns expectation values of Pauli
Z observables.

The module is intentionally lightweight and fully compatible with the
Aer simulator; it can be extended with real quantum backends.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Sequence, List
from dataclasses import dataclass

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class HybridSelfAttentionQuantum:
    def __init__(self,
                 n_qubits: int,
                 graph_threshold: float = 0.8) -> None:
        self.n_qubits = n_qubits
        self.graph_threshold = graph_threshold
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_unitary_circuit(self,
                               rotation_params: np.ndarray,
                               entangle_params: np.ndarray,
                               adjacency: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[i], self.qr[i])
            qc.ry(entangle_params[i], self.qr[i])
            qc.rz(rotation_params[self.n_qubits + i], self.qr[i])
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[self.n_qubits + i], self.qr[i], self.qr[i + 1])
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                if adjacency[i, j] >= self.graph_threshold:
                    qc.cz(self.qr[i], self.qr[j])
        return qc

    def _build_measure_circuit(self,
                               rotation_params: np.ndarray,
                               entangle_params: np.ndarray,
                               adjacency: np.ndarray) -> QuantumCircuit:
        qc = self._build_unitary_circuit(rotation_params, entangle_params, adjacency)
        qc.measure(self.qr, self.cr)
        return qc

    def run(self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> dict:
        adjacency = np.ones((self.n_qubits, self.n_qubits)) - np.eye(self.n_qubits)
        qc = self._build_measure_circuit(rotation_params, entangle_params, adjacency)
        job = execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        adjacency = np.ones((self.n_qubits, self.n_qubits)) - np.eye(self.n_qubits)
        for params in parameter_sets:
            total = len(params)
            half = total // 2
            rotation_params = np.array(params[:half])
            entangle_params = np.array(params[half:])
            qc = self._build_unitary_circuit(rotation_params, entangle_params, adjacency)
            state = Statevector.from_instruction(qc)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = ["HybridSelfAttentionQuantum", "FraudLayerParameters"]
