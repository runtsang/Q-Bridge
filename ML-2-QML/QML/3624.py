"""Quantum implementation of QCNN fused with a self‑attention sub‑circuit."""

import math
import numpy as np
from qiskit import QuantumCircuit, ParameterVector, Aer
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution sub‑circuit."""
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
    """Two‑qubit pooling sub‑circuit."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def attention_circuit(params_rot: ParameterVector,
                      params_ent: ParameterVector,
                      qubits: list[int]) -> QuantumCircuit:
    """Self‑attention inspired sub‑circuit."""
    qc = QuantumCircuit(len(qubits))
    # Rotation stage
    for i, q in enumerate(qubits):
        qc.rx(params_rot[3 * i], q)
        qc.ry(params_rot[3 * i + 1], q)
        qc.rz(params_rot[3 * i + 2], q)
    # Entanglement stage
    for i in range(len(qubits) - 1):
        qc.crx(params_ent[i], qubits[i], qubits[i + 1])
    return qc


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Layer of pairwise convolutions."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for i in range(0, num_qubits, 2):
        sub = conv_circuit(params[idx:idx + 3])
        qc.append(sub, [i, i + 1])
        qc.barrier()
        idx += 3
    return qc


def pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Layer of pairwise pooling operations."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for i in range(0, num_qubits, 2):
        sub = pool_circuit(params[idx:idx + 3])
        qc.append(sub, [i, i + 1])
        qc.barrier()
        idx += 3
    return qc


def attention_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Self‑attention layer embedding rotation and entanglement."""
    rot = ParameterVector(param_prefix + "_rot", length=num_qubits * 3)
    ent = ParameterVector(param_prefix + "_ent", length=num_qubits - 1)
    qubits = list(range(num_qubits))
    return attention_circuit(rot, ent, qubits)


def QCNN() -> EstimatorQNN:
    """Factory returning a quantum neural network with convolution + attention."""
    estimator = StatevectorEstimator()

    # Feature map
    feature_map = ZFeatureMap(8)
    feature_map = feature_map.decompose()

    # Ansatz construction
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First convolution and pooling
    ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(pool_layer(8, "p1"), range(8), inplace=True)

    # Attention after first pooling
    ansatz.compose(attention_layer(8, "att1"), range(8), inplace=True)

    # Second convolution and pooling
    ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(pool_layer(4, "p2"), range(4, 8), inplace=True)

    # Attention after second pooling
    ansatz.compose(attention_layer(4, "att2"), range(4, 8), inplace=True)

    # Third convolution and pooling
    ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(pool_layer(2, "p3"), range(6, 8), inplace=True)

    # Full circuit
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

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
