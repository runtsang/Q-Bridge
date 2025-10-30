"""
Quantum neural network that mirrors the classical HybridEstimatorQNN.
The circuit consists of a ZFeatureMap and a QCNN‑style ansatz built
with convolution and pooling layers.  The function returns a
qiskit_machine_learning.neural_networks.EstimatorQNN instance
ready for training with a StatevectorEstimator.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def _single_conv_circuit(params) -> QuantumCircuit:
    sub = QuantumCircuit(2)
    sub.rz(-np.pi / 2, 1)
    sub.cx(1, 0)
    sub.rz(params[0], 0)
    sub.ry(params[1], 1)
    sub.cx(0, 1)
    sub.ry(params[2], 1)
    sub.cx(1, 0)
    sub.rz(np.pi / 2, 0)
    return sub

def _single_pool_circuit(params) -> QuantumCircuit:
    sub = QuantumCircuit(2)
    sub.rz(-np.pi / 2, 1)
    sub.cx(1, 0)
    sub.rz(params[0], 0)
    sub.ry(params[1], 1)
    sub.cx(0, 1)
    sub.ry(params[2], 1)
    return sub

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        conv = _single_conv_circuit(params[param_index:param_index+3])
        qc.append(conv, [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        conv = _single_conv_circuit(params[param_index:param_index+3])
        qc.append(conv, [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def _pool_layer(sources, sinks, param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    param_index = 0
    for src, snk in zip(sources, sinks):
        pool = _single_pool_circuit(params[param_index:param_index+3])
        qc.append(pool, [src, snk])
        qc.barrier()
        param_index += 3
    return qc

def HybridEstimatorQNN() -> EstimatorQNN:
    """
    Build a QCNN‑style EstimatorQNN that can be trained with a classical
    optimizer and used as a drop‑in replacement for the original
    EstimatorQNN.  The ansatz consists of three convolutional and
    pooling stages followed by a Z feature map.  The observable is a
    single Z on the first qubit.
    """
    estimator = StatevectorEstimator()
    feature_map = ZFeatureMap(8)
    # Build ansatz
    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(_conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(_pool_layer([0,1,2,3], [4,5,6,7], "p1"), list(range(8)), inplace=True)
    ansatz.compose(_conv_layer(4, "c2"), list(range(4,8)), inplace=True)
    ansatz.compose(_pool_layer([0,1], [2,3], "p2"), list(range(4,8)), inplace=True)
    ansatz.compose(_conv_layer(2, "c3"), list(range(6,8)), inplace=True)
    ansatz.compose(_pool_layer([0], [1], "p3"), list(range(6,8)), inplace=True)
    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    return EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

__all__ = ["HybridEstimatorQNN"]
