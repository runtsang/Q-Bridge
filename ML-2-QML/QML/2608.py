"""Quantum hybrid QCNN with a per‑patch quantum filter.

The circuit encodes the MNIST image into 8 qubits using a ZFeatureMap,
then applies a sequence of convolutional and pooling layers inspired
by the QCNN helper.  Each convolutional layer consists of a
parameterised two‑qubit block that acts on neighbouring qubit pairs,
while the pooling layers reduce the qubit count by discarding the
second qubit of each pair.  The resulting circuit is wrapped in an
EstimatorQNN so it can be used as a differentiable quantum neural
network.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Parameterised 2‑qubit convolution block."""
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
    """Parameterised 2‑qubit pooling block."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolutional layer that applies _conv_circuit to each adjacent pair."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        sub = _conv_circuit(params[i // 2 * 3 : i // 2 * 3 + 3])
        qc.append(sub, [i, i + 1])
    return qc


def _pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Pooling layer that applies _pool_circuit to each adjacent pair."""
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        sub = _pool_circuit(params[i // 2 * 3 : i // 2 * 3 + 3])
        qc.append(sub, [i, i + 1])
    return qc


def QuanvolutionQCNNQNN() -> EstimatorQNN:
    """Constructs the hybrid QCNN quantum neural network."""
    estimator = StatevectorEstimator()

    # Feature map: encode each pixel into a qubit
    feature_map = ZFeatureMap(8)

    # Ansatz: three conv–pool stages
    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(_conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(_pool_layer(8, "p1"), range(8), inplace=True)
    ansatz.compose(_conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(_pool_layer(4, "p2"), range(4, 8), inplace=True)
    ansatz.compose(_conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(_pool_layer(2, "p3"), range(6, 8), inplace=True)

    # Combine feature map and ansatz
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


__all__ = ["QuanvolutionQCNNQNN"]
