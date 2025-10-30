"""Quantum circuit factory and sampler for hybrid models."""
from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Return a variational circuit, encoding and weight parameters, and measurement observables."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for q, param in zip(range(num_qubits), encoding):
        qc.rx(param, q)

    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables


def build_sampler_qnn(num_qubits: int = 2) -> QSamplerQNN:
    """Return a Qiskit SamplerQNN that can be used as a feature extractor."""
    inputs = ParameterVector("input", num_qubits)
    weights = ParameterVector("weight", 4)
    qc = QuantumCircuit(num_qubits)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)

    sampler = StatevectorSampler()
    sampler_qnn = QSamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)
    return sampler_qnn


__all__ = ["build_classifier_circuit", "build_sampler_qnn"]
