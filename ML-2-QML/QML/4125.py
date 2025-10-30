"""Quantum implementation of the hybrid classifier with a variational ansatz."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

from.FastBaseEstimator import FastBaseEstimator

class HybridQuantumClassifier:
    """Quantum circuit factory and evaluator that mirrors the classical interface."""
    def __init__(self, num_qubits: int, depth: int = 3):
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.params, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, List, List, List[SparsePauliOp]]:
        """Create a layered ansatz with explicit encoding and entangling gates."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        for i, p in enumerate(encoding):
            qc.rx(p, i)

        idx = 0
        for _ in range(self.depth):
            for i in range(self.num_qubits):
                qc.ry(weights[idx], i)
                idx += 1
            for i in range(self.num_qubits - 1):
                qc.cz(i, i + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return qc, list(encoding), list(weights), observables

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: List[List[float]],
    ) -> List[List[complex]]:
        """Delegate to the lightweight FastBaseEstimator."""
        estimator = FastBaseEstimator(self.circuit)
        return estimator.evaluate(observables, parameter_sets)

    def predict(self, state_batch: np.ndarray) -> np.ndarray:
        """Classical post‑processing of quantum expectations into class labels."""
        bsz = state_batch.shape[0]
        exp_vals = np.zeros((bsz, self.num_qubits), dtype=complex)
        for i, state in enumerate(state_batch):
            circ = self.circuit.assign_parameters(state.tolist(), inplace=False)
            sv = Statevector.from_instruction(circ)
            for j, obs in enumerate(self.observables):
                exp_vals[i, j] = sv.expectation_value(obs)
        logits = exp_vals.mean(axis=1)
        return (logits > 0).astype(int)

__all__ = ["HybridQuantumClassifier"]
