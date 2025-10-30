"""Quantum implementation of the QCNN ansatz with depth scaling and pooling."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as EstimatorQ
from qiskit_machine_learning.neural_networks import EstimatorQNN

def conv_circuit(params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
    """Two‑qubit convolution unitary."""
    qc = QuantumCircuit(len(qubits))
    qc.rz(-np.pi / 2, qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(params[0], qubits[0])
    qc.ry(params[1], qubits[1])
    qc.cx(qubits[0], qubits[1])
    qc.ry(params[2], qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(np.pi / 2, qubits[0])
    return qc

def pool_circuit(params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
    """Two‑qubit pooling unitary."""
    qc = QuantumCircuit(len(qubits))
    qc.rz(-np.pi / 2, qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(params[0], qubits[0])
    qc.ry(params[1], qubits[1])
    qc.cx(qubits[0], qubits[1])
    qc.ry(params[2], qubits[1])
    return qc

def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Layer that applies conv_circuit to adjacent pairs."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
    idx = 0
    for i in range(0, num_qubits, 2):
        sub = conv_circuit(params[idx:idx+3], [i, i+1])
        qc.append(sub.to_instruction(), [i, i+1])
        idx += 3
    return qc

def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Layer that applies pool_circuit to adjacent pairs."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
    idx = 0
    for i in range(0, num_qubits, 2):
        sub = pool_circuit(params[idx:idx+3], [i, i+1])
        qc.append(sub.to_instruction(), [i, i+1])
        idx += 3
    return qc

def QCNN() -> EstimatorQNN:
    """Builds a depth‑scaled QCNN ansatz and returns an EstimatorQNN."""
    estimator = EstimatorQ()

    # Feature map for 8 input features
    feature_map = ZFeatureMap(8)

    # Build ansatz with 3 convolution–pool pairs
    ansatz = QuantumCircuit(8)
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer(8, "p1"), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer(4, "p2"), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer(2, "p3"), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNN"]
