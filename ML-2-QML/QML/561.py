"""Quantum convolutional neural network with configurable depth.

The quantum ansatz is constructed from repeated convolution and
pooling blocks.  The depth of the network can be tuned via the
``depth`` argument, allowing a direct comparison with the classical
counterpart.  A single observable (Z‑measurement on all qubits) is
used for the output.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution block."""
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


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling block."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Wraps ``conv_circuit`` across disjoint qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.append(conv_circuit(params[idx : idx + 3]), [q1, q2])
        qc.barrier()
        idx += 3
    return qc


def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Wraps ``pool_circuit`` across disjoint qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.append(pool_circuit(params[idx : idx + 3]), [q1, q2])
        qc.barrier()
        idx += 3
    return qc


def build_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    """Construct a convolutional‑pooling ansatz of the requested depth."""
    qc = QuantumCircuit(num_qubits)
    for d in range(depth):
        qc.append(conv_layer(num_qubits, f"c{d+1}"), range(num_qubits))
        qc.append(pool_layer(num_qubits, f"p{d+1}"), range(num_qubits))
    return qc


def QCNNExtendedQNN(depth: int = 3, num_qubits: int = 8) -> EstimatorQNN:
    """Return a hybrid QNN with the specified depth."""
    feature_map = ZFeatureMap(num_qubits)
    ansatz = build_ansatz(num_qubits, depth)

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)

    observable = SparsePauliOp.from_list([("Z" * num_qubits, 1)])
    estimator = Estimator()

    return EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )


__all__ = ["QCNNExtendedQNN"]
