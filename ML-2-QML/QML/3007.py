"""Hybrid quantum fully connected layer with estimator utilities.

The circuit implements a parameterised Ry rotation on each qubit
preceded by Hadamards. It supports evaluation of arbitrary
observables via the Statevector simulator and optional shot noise.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List, Iterable

class HybridFCL:
    """Quantum hybrid fully‑connected layer with estimator support."""

    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 100) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        theta = Parameter("theta")
        qc.h(range(self.n_qubits))
        qc.barrier()
        qc.ry(theta, range(self.n_qubits))
        qc.measure_all()
        self._theta = theta
        return qc

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit for each theta and return expectation value."""
        jobs = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self._theta: theta} for theta in thetas],
        )
        result = jobs.result()
        counts = result.get_counts(self._circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each observable and parameter set."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._circuit.assign_parameters(dict(zip(self._circuit.parameters, values)), inplace=False)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results
