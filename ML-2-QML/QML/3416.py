"""Hybrid quantum self‑attention + QCNN variational circuit."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp

__all__ = ["HybridQuantumAttentionQCNN"]


def _conv_circuit(params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
    """Two‑qubit convolution subcircuit used by all conv layers."""
    qc = QuantumCircuit(len(qubits))
    for i, q in enumerate(qubits):
        qc.rz(-np.pi / 2, q)
    for i in range(0, len(qubits) - 1, 2):
        qc.cx(qubits[i + 1], qubits[i])
    qc.rz(params[0], qubits[0])
    qc.ry(params[1], qubits[1])
    qc.cx(qubits[0], qubits[1])
    qc.ry(params[2], qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(np.pi / 2, qubits[0])
    return qc


def _pool_circuit(params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
    """Two‑qubit pooling subcircuit."""
    qc = QuantumCircuit(len(qubits))
    qc.rz(-np.pi / 2, qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(params[0], qubits[0])
    qc.ry(params[1], qubits[1])
    qc.cx(qubits[0], qubits[1])
    qc.ry(params[2], qubits[1])
    return qc


def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Convolutional layer composed of consecutive 2‑qubit conv blocks."""
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(f"{prefix}_conv", length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        block = _conv_circuit(params[param_index:param_index + 3], [i, i + 1])
        qc.append(block, [i, i + 1])
        param_index += 3
    return qc


def _pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Pooling layer composed of consecutive 2‑qubit pool blocks."""
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(f"{prefix}_pool", length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        block = _pool_circuit(params[param_index:param_index + 3], [i, i + 1])
        qc.append(block, [i, i + 1])
        param_index += 3
    return qc


def _self_attention_block(num_qubits: int) -> QuantumCircuit:
    """Quantum self‑attention subcircuit parameterized by rotation and
    controlled‑RX entangling gates."""
    qc = QuantumCircuit(num_qubits)
    rot_params = ParameterVector("rot", length=num_qubits * 3)
    ent_params = ParameterVector("ent", length=num_qubits - 1)

    for q in range(num_qubits):
        qc.rx(rot_params[3 * q], q)
        qc.ry(rot_params[3 * q + 1], q)
        qc.rz(rot_params[3 * q + 2], q)

    for q in range(num_qubits - 1):
        qc.crx(ent_params[q], q, q + 1)

    return qc


def HybridQuantumAttentionQCNN() -> EstimatorQNN:
    """
    Builds a variational quantum circuit that concatenates a feature map,
    a quantum self‑attention block, and a QCNN‑style convolution/pooling stack.
    The returned :class:`EstimatorQNN` can be trained with standard optimizers.
    """
    n_qubits = 8
    feature_map = ZFeatureMap(n_qubits)
    circuit = QuantumCircuit(n_qubits)

    # Encode classical data
    circuit.compose(feature_map, range(n_qubits), inplace=True)

    # Quantum self‑attention
    circuit.compose(_self_attention_block(n_qubits), range(n_qubits), inplace=True)

    # QCNN layers
    circuit.compose(_conv_layer(n_qubits, "c1"), range(n_qubits), inplace=True)
    circuit.compose(_pool_layer(n_qubits, "p1"), range(n_qubits), inplace=True)
    circuit.compose(_conv_layer(n_qubits // 2, "c2"), range(n_qubits // 2), inplace=True)
    circuit.compose(_pool_layer(n_qubits // 4, "p2"), range(n_qubits // 4), inplace=True)
    circuit.compose(_conv_layer(n_qubits // 8, "c3"), range(n_qubits // 8), inplace=True)
    circuit.compose(_pool_layer(n_qubits // 16, "p3"), range(n_qubits // 16), inplace=True)

    # Observable for a single‑qubit Z measurement
    observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])

    # Variational ansatz
    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=circuit.parameters,
        estimator=estimator,
    )
    return qnn
