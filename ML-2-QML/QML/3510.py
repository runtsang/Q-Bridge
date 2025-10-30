"""Quantum components for the hybrid QCNN architecture."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN

def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unit, parameterized by 3 angles."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc

def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Builds a convolutional layer acting on pairs of qubits."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.append(conv_circuit(params[i:i+3]), [i, i+1])
    for i in range(1, num_qubits-1, 2):
        qc.append(conv_circuit(params[i:i+3]), [i, i+1])
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unit."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
    """Reduces the qubit count by measuring pairs of source‑sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(prefix, length=len(sources) * 3)
    for src, snk in zip(sources, sinks):
        qc.append(pool_circuit(params[:3]), [src, snk])
        params = params[3:]
    return qc

def QCNNQuantum() -> EstimatorQNN:
    """Constructs the full QCNN ansatz and returns an EstimatorQNN module."""
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="Ansatz")

    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    circuit.decompose(inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    estimator = StatevectorEstimator()
    return EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

def SamplerQNN() -> SamplerQNN:
    """A lightweight quantum sampler that returns a probability vector."""
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

    sampler = StatevectorSampler()
    return SamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)

__all__ = ["QCNNQuantum", "SamplerQNN"]
