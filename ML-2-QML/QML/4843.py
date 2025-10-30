"""Quantum neural network mirroring the QCNN architecture for regression."""
from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QNNClass
from qiskit_machine_learning.utils import algorithm_globals


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
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


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.append(_conv_circuit(params[idx : idx + 3]), [q1, q2])
        idx += 3
    return qc


def _pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.append(_pool_circuit(params[idx : idx + 3]), [q1, q2])
        idx += 3
    return qc


def EstimatorQNN() -> QNNClass:
    algorithm_globals.random_seed = 12345
    estimator = StatevectorEstimator()

    feature_map = ZFeatureMap(8)

    # Build the ansatz following the QCNN pattern
    ansatz = QuantumCircuit(8)
    ansatz.compose(_conv_layer(8, "c1"), inplace=True)
    ansatz.compose(_pool_layer(8, "p1"), inplace=True)
    ansatz.compose(_conv_layer(4, "c2"), inplace=True)
    ansatz.compose(_pool_layer(4, "p2"), inplace=True)
    ansatz.compose(_conv_layer(2, "c3"), inplace=True)
    ansatz.compose(_pool_layer(2, "p3"), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Observable for regression: Z on the first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = QNNClass(
        circuit=circuit,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["EstimatorQNN"]
