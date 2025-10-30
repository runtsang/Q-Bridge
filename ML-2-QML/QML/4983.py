"""QCNN__gen244.py – Quantum variant built with Qiskit."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def _conv_block(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Two‑qubit convolution unit with 3 parameters per qubit pair."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.rz(-np.pi / 2, i + 1)
        qc.cx(i + 1, i)
        qc.rz(params[i * 3], i)
        qc.ry(params[i * 3 + 1], i + 1)
        qc.cx(i, i + 1)
        qc.ry(params[i * 3 + 2], i + 1)
        qc.cx(i + 1, i)
        qc.rz(np.pi / 2, i)
    return qc


def _pool_block(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Two‑qubit pooling unit with 3 parameters per qubit pair."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.rz(-np.pi / 2, i + 1)
        qc.cx(i + 1, i)
        qc.rz(params[i * 3], i)
        qc.ry(params[i * 3 + 1], i + 1)
        qc.cx(i, i + 1)
        qc.ry(params[i * 3 + 2], i + 1)
    return qc


def _self_attention_block(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Quantum self‑attention sub‑circuit using CRX entanglement."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(num_qubits):
        qc.rz(params[i * 3], i)
        qc.ry(params[i * 3 + 1], i)
        qc.rx(params[i * 3 + 2], i)
    for i in range(num_qubits - 1):
        qc.crx(params[i + num_qubits * 3], i, i + 1)
    return qc


def _fully_connected_block(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Parameterised single‑qubit rotation followed by measurement."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits)
    for i in range(num_qubits):
        qc.ry(params[i], i)
    return qc


def QCNN__gen244() -> EstimatorQNN:
    """Factory returning the Quantized QCNN with self‑attention and FC layers."""
    # Feature map
    feature_map = ZFeatureMap(8)
    # Convolution and pooling stages
    conv1 = _conv_block(8, "c1")
    pool1 = _pool_block(8, "p1")
    conv2 = _conv_block(4, "c2")
    pool2 = _pool_block(4, "p2")
    conv3 = _conv_block(2, "c3")
    pool3 = _pool_block(2, "p3")

    # Self‑attention sub‑circuit
    attention = _self_attention_block(4, "a")

    # Fully‑connected measurement block
    fc_block = _fully_connected_block(1, "fc")

    # Assemble ansatz
    ansatz = QuantumCircuit(8)
    ansatz.append(conv1, range(8))
    ansatz.append(pool1, range(8))
    ansatz.append(conv2, range(4, 8))
    ansatz.append(pool2, range(4, 8))
    ansatz.append(conv3, range(6, 8))
    ansatz.append(pool3, range(6, 8))
    ansatz.append(attention, range(0, 4))          # attention on first 4 qubits
    ansatz.append(fc_block, range(0, 1))           # final FC on first qubit

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.append(feature_map, range(8))
    circuit.append(ansatz, range(8))

    # Observable: expectation of Z on the first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    estimator = StatevectorEstimator()

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["QCNN__gen244", "QCNN__gen244"]
