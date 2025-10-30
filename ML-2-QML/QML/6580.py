"""Hybrid quantum estimator that evaluates expectation values of observables
for a parametrized quantum circuit and optionally adds shot noise.

It extends the original FastBaseEstimator by allowing a convolutional
quantum filter (QuanvCircuit) that can be used as a preprocessing layer.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit import execute, Aer
from collections.abc import Iterable, Sequence
from typing import List, Optional


class QuanvCircuit:
    """Quantum convolution filter used as a preprocessing layer."""
    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 1024, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, depth=2)
        self._circuit.measure_all()

    def run(self, data):
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {theta: np.pi if val > self.threshold else 0 for theta, val in zip(self.theta, dat)}
            param_binds.append(bind)
        job = execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)


class HybridEstimator:
    """Hybrid quantum estimator that evaluates expectation values of observables
    given a parametrized circuit. Supports optional shot noise and a quantum
    convolution filter as part of the circuit."""
    def __init__(self, circuit: QuantumCircuit, conv: Optional[QuanvCircuit] = None, shots: int | None = None) -> None:
        self.circuit = circuit
        self.conv = conv
        self.shots = shots or 1024
        self.backend = Aer.get_backend("qasm_simulator")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[List[complex]]:
        if shots is None:
            shots = self.shots
        results: List[List[complex]] = []
        for values in parameter_sets:
            if self.conv:
                conv_out = self.conv.run(values)
                bound_values = list(values) + [conv_out]
            else:
                bound_values = values
            circ = self._bind(bound_values)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        if shots is not None and shots < 1e6:
            rng = np.random.default_rng()
            noisy = []
            for row in results:
                noisy_row = [float(rng.normal(val.real, max(1e-6, 1 / shots))) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots)) for val in row]
                noisy.append(noisy_row)
            return noisy
        return results


__all__ = ["HybridEstimator", "QuanvCircuit"]
