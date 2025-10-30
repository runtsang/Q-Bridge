"""Quantum kernel and estimator utilities using Qiskit."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class KernalAnsatz:
    """Parameterized circuit that encodes classical data into a quantum state."""
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.base_circuit = QuantumCircuit(n_qubits)

    def encode(self, x: Sequence[float]) -> QuantumCircuit:
        circ = self.base_circuit.copy()
        for i, val in enumerate(x):
            circ.ry(val, i)
        return circ

    def encode_diff(self, x: Sequence[float], y: Sequence[float]) -> QuantumCircuit:
        circ = self.base_circuit.copy()
        for i, val in enumerate(x):
            circ.ry(val, i)
        for i, val in enumerate(y):
            circ.ry(-val, i)
        return circ

class Kernel:
    """Evaluates the quantum kernel via overlap of encoded states."""
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.ansatz = KernalAnsatz(n_qubits)

    def __call__(self, x: Sequence[float], y: Sequence[float]) -> float:
        circ = self.ansatz.encode_diff(x, y)
        sv = Statevector.from_instruction(circ)
        return float(np.abs(sv[0]) ** 2)

def kernel_matrix(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y) for y in b] for x in a])

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to the estimator."""
    def evaluate(self, observables, parameter_sets, *, shots: int | None = None, seed: int | None = None):
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy.append([complex(rng.normal(val.real, max(1e-6, 1 / shots)),
                                  rng.normal(val.imag, max(1e-6, 1 / shots))) for val in row])
        return noisy

class QuantumKernelMethod:
    """High-level API combining quantum kernel and fast estimator."""
    def __init__(self, n_qubits: int = 4, circuit: QuantumCircuit | None = None):
        self.kernel = Kernel(n_qubits)
        self.estimator = FastEstimator(circuit) if circuit else None

    def kernel_matrix(self, a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> np.ndarray:
        return kernel_matrix(a, b)

    def evaluate(self, observables, parameter_sets, *, shots: int | None = None, seed: int | None = None):
        if self.estimator is None:
            raise ValueError("No circuit supplied for evaluation.")
        return self.estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "FastBaseEstimator", "FastEstimator", "QuantumKernelMethod"]
