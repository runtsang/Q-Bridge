"""Quantum variational circuit that consumes the angles produced by the
HybridEstimator class.  It returns expectation values of Pauli‑Z on each
qubit, suitable for hybrid training or evaluation."""
from __future__ import annotations

from typing import Sequence, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp

class HybridEstimator:
    """Parameterized 4‑qubit ansatz with entanglement."""
    def __init__(self) -> None:
        self.n_qubits = 4
        # parameters θ₀…θ₃
        self.params = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        # base circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        # create a uniform superposition
        for q in range(self.n_qubits):
            self.circuit.h(q)
        # parameterised rotations
        for q, p in enumerate(self.params):
            self.circuit.ry(p, q)
        # entangling layer (CRX gates)
        for q in range(self.n_qubits - 1):
            self.circuit.crx(np.pi / 4, q, q + 1)
        # observables: Pauli‑Z on each qubit
        self.observables = [
            SparsePauliOp.from_list([(f"{'I'*i}Z{'I'*(self.n_qubits-i-1)}", 1)])
            for i in range(self.n_qubits)
        ]

    def bind(self, theta: Sequence[float]) -> QuantumCircuit:
        """Return a circuit with θ values bound."""
        if len(theta)!= self.n_qubits:
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = {p: t for p, t in zip(self.params, theta)}
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """
        Compute expectation values of the stored observables for every
        parameter set.

        Parameters
        ----------
        parameter_sets : Sequence[Sequence[float]]
            Iterable of length‑4 angle lists.

        Returns
        -------
        List[List[complex]]
            Nested list of expectation values (one list per set).
        """
        results: List[List[complex]] = []
        for theta in parameter_sets:
            circ = self.bind(theta)
            state = Statevector.from_instruction(circ)
            row: List[complex] = [state.expectation_value(obs) for obs in self.observables]
            results.append(row)
        return results

__all__ = ["HybridEstimator"]
