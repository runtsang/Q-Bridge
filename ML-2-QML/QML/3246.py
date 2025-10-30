"""
HybridSamplerQNN: Quantum circuit that implements a QCNN‑style ansatz and samples its output.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as EstimatorPrimitive
from qiskit_machine_learning.neural_networks import EstimatorQNN


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Two‑qubit convolution block used in the QCNN ansatz.
    """
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
    """
    Two‑qubit pooling block used in the QCNN ansatz.
    """
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """
    Builds a convolution layer by applying the two‑qubit conv block
    across adjacent qubit pairs.
    """
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        block = _conv_circuit(params[i * 3 : (i + 2) * 3])
        qc.append(block.to_instruction(), [i, i + 1])
    return qc


def _pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """
    Builds a pooling layer by applying the two‑qubit pool block
    across adjacent qubit pairs.
    """
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        block = _pool_circuit(params[i * 3 : (i + 2) * 3])
        qc.append(block.to_instruction(), [i, i + 1])
    return qc


def HybridSamplerQNN() -> EstimatorQNN:
    """
    Constructs a QCNN‑style quantum neural network with a ZFeatureMap feature
    embedding, multiple convolution and pooling layers, and a single‑qubit
    observable for classification. The circuit is passed to an EstimatorQNN
    which behaves like a torch.nn.Module for end‑to‑end training.
    """
    # Feature map that encodes the input data into a quantum state
    feature_map = ZFeatureMap(8)
    input_params = feature_map.parameters

    # QCNN ansatz: three conv–pool stages reducing qubit count
    ansatz = QuantumCircuit(8, name="QCNN_Ansatz")

    # Stage 1: conv (8 qubits) -> pool (4 qubits)
    ansatz.compose(_conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(_pool_layer(8, "p1"), range(8), inplace=True)

    # Stage 2: conv (4 qubits) -> pool (2 qubits)
    ansatz.compose(_conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(_pool_layer(4, "p2"), range(4, 8), inplace=True)

    # Stage 3: conv (2 qubits) -> pool (1 qubit)
    ansatz.compose(_conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(_pool_layer(2, "p3"), range(6, 8), inplace=True)

    # Combine feature map and ansatz into a single circuit
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Observable for binary classification (single‑qubit Z on ancilla)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Estimator primitive for expectation value evaluation
    estimator = EstimatorPrimitive()

    # Wrap the circuit as a neural network layer
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=input_params,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["HybridSamplerQNN"]
