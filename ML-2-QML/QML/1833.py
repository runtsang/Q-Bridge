"""
Quantum Convolutional Neural Network (QCNN) implementation using Qiskit.

This module builds a reusable QCNN circuit with parameter‑sharing and
provides a helper to construct the full 8‑qubit circuit, feature map,
and EstimatorQNN wrapper.  The design mirrors the classical
architecture and allows easy integration into hybrid workflows.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA


def _conv_block(qubits, params):
    """Two‑qubit convolution block used by all conv layers."""
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


def conv_layer(num_qubits, prefix):
    """Build a convolution layer with parameter sharing."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
    for i in range(0, num_qubits, 2):
        block = _conv_block([i, i + 1], params[i // 2 * 3 : (i // 2 + 1) * 3])
        qc.append(block, [i, i + 1])
    return qc


def _pool_block(qubits, params):
    """Two‑qubit pooling block used by all pool layers."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def pool_layer(sources, sinks, prefix):
    """Build a pooling layer that maps source qubits to sink qubits."""
    qc = QuantumCircuit(len(sources) + len(sinks))
    params = ParameterVector(prefix, length=len(sources) * 3)
    for src, snk, idx in zip(sources, sinks, range(len(sources))):
        block = _pool_block([src, snk], params[idx * 3 : (idx + 1) * 3])
        qc.append(block, [src, snk])
    return qc


def build_qcnn(num_qubits: int = 8):
    """
    Construct the full QCNN circuit with feature map, conv, and pool layers.

    Args:
        num_qubits: Total number of qubits in the QCNN (must be a power of two).

    Returns:
        circuit: The composed QCNN circuit ready for training.
        feature_map: The ZFeatureMap instance used for data encoding.
    """
    if num_qubits not in {4, 8, 16}:
        raise ValueError("Supported qubit counts: 4, 8, 16.")
    feature_map = ZFeatureMap(num_qubits)
    circuit = QuantumCircuit(num_qubits)

    # Feature encoding
    circuit.compose(feature_map, range(num_qubits), inplace=True)

    # First conv / pool
    circuit.compose(conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
    circuit.compose(
        pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), "p1"),
        range(num_qubits),
        inplace=True,
    )

    # Second conv / pool
    circuit.compose(conv_layer(num_qubits // 2, "c2"), range(num_qubits // 2, num_qubits), inplace=True)
    circuit.compose(
        pool_layer([0, 1], [2, 3], "p2"),
        range(num_qubits // 2, num_qubits),
        inplace=True,
    )

    # Third conv / pool
    circuit.compose(conv_layer(num_qubits // 4, "c3"), range(num_qubits // 2, num_qubits), inplace=True)
    circuit.compose(
        pool_layer([0], [1], "p3"),
        range(num_qubits // 2, num_qubits),
        inplace=True,
    )

    return circuit, feature_map


def QCNN():
    """
    Factory returning an EstimatorQNN suitable for hybrid training.

    The returned QNN uses a sparse Pauli observable on the first qubit.
    """
    estimator = Estimator()
    circuit, feature_map = build_qcnn(8)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=circuit.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["QCNN", "build_qcnn"]
