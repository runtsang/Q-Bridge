"""Hybrid quantum estimator that unifies a generic quantum circuit with a
self‑attention style block.

The estimator keeps the same lightweight API as the original FastBaseEstimator
while providing a built‑in QuantumSelfAttention circuit.  It can evaluate
expectation values of arbitrary BaseOperator observables or raw measurement
outcomes.  A subclass with shot noise is provided for consistency with the
classical side.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class QuantumSelfAttention:
    """Quantum circuit that implements a self‑attention style block.

    The circuit consists of rotation gates on each qubit followed by
    controlled‑RX entangling gates.  Parameters are supplied as flat arrays
    and reshaped to match the gate schedule.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

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
    ) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

class HybridEstimator:
    """Evaluate a quantum circuit or a QuantumSelfAttention instance.

    The estimator accepts a QuantumCircuit or a QuantumSelfAttention object.
    It evaluates expectation values of BaseOperator observables or raw
    measurement distributions.  The public API mirrors the classical
    counterpart for consistency.
    """
    def __init__(self, circuit: QuantumCircuit | QuantumSelfAttention) -> None:
        self.circuit = circuit
        if isinstance(circuit, QuantumSelfAttention):
            self._total_params = 3 * circuit.n_qubits + (circuit.n_qubits - 1)
        else:
            self._total_params = len(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= self._total_params:
            raise ValueError("Parameter count mismatch for bound circuit.")
        if isinstance(self.circuit, QuantumSelfAttention):
            rotation_len = 3 * self.circuit.n_qubits
            rotation_params = np.array(parameter_values[:rotation_len])
            entangle_params = np.array(parameter_values[rotation_len:])
            return self.circuit._build_circuit(rotation_params, entangle_params)
        else:
            mapping = dict(zip(self.circuit.parameters, parameter_values))
            return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            circuit = self._bind(values)
            state = Statevector.from_instruction(circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class HybridEstimatorWithShots(HybridEstimator):
    """Same as HybridEstimator but uses a finite number of shots and returns
    expectation values derived from the measurement counts.  This simulates
    shot noise in a quantum experiment.
    """
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int = 1024,
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        backend = Aer.get_backend("qasm_simulator")
        for values in parameter_sets:
            if isinstance(self.circuit, QuantumSelfAttention):
                counts = self.circuit.run(backend, *values, shots=shots)
            else:
                circuit = self._bind(values)
                job = execute(circuit, backend, shots=shots)
                counts = job.result().get_counts(circuit)
            state = Statevector.from_counts(counts)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

def simple_self_attention_circuit(n_qubits: int = 4) -> QuantumSelfAttention:
    """Convenience factory that creates a bare QuantumSelfAttention instance."""
    return QuantumSelfAttention(n_qubits=n_qubits)

__all__ = ["HybridEstimator", "HybridEstimatorWithShots", "QuantumSelfAttention", "simple_self_attention_circuit"]
