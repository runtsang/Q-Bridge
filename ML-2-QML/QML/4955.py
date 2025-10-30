"""Hybrid estimator that works with Qiskit quantum circuits.

Features
--------
* deterministic statevector evaluation or shot‑based sampling
* optional self‑attention subcircuit (parameterized rotation and entangling gates)
* convenient factory functions for EstimatorQNN, SamplerQNN, and SelfAttention
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import StatevectorEstimator

class QuantumSelfAttention:
    """Builds a self‑attention style block with rotation and controlled‑RX gates."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
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
    """Hybrid estimator that can evaluate a quantum circuit with optional shot sampling."""
    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        self_attention: bool = False,
    ) -> None:
        self.circuit = circuit
        self.shots = shots
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.backend = Aer.get_backend("statevector_simulator")
        if self_attention:
            # Attach a self‑attention subcircuit with random parameters
            n_qubits = circuit.num_qubits
            sa = QuantumSelfAttention(n_qubits)
            rot_params = self.rng.uniform(0, 2 * np.pi, size=3 * n_qubits)
            ent_params = self.rng.uniform(0, 2 * np.pi, size=n_qubits - 1)
            subcircuit = sa._build_circuit(rot_params, ent_params)
            self.circuit.compose(subcircuit, inplace=True)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []

        estimator = StatevectorEstimator()
        for params in parameter_sets:
            bound_circ = self._bind(params)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if self.shots is not None:
            noisy: List[List[complex]] = []
            std = max(1e-6, 1 / np.sqrt(self.shots))
            for row in results:
                noisy_row = [complex(self.rng.normal(val.real, std) + 1j * self.rng.normal(val.imag, std)) for val in row]
                noisy.append(noisy_row)
            return noisy

        return results


def EstimatorQNN() -> QuantumCircuit:
    """Quantum version of the EstimatorQNN example."""
    params = [Parameter("input1"), Parameter("weight1")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)
    return qc


def SamplerQNN() -> QuantumCircuit:
    """Quantum version of the SamplerQNN example."""
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    return qc

__all__ = ["HybridEstimator", "QuantumSelfAttention", "EstimatorQNN", "SamplerQNN"]
