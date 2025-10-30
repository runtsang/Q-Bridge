"""Hybrid QCNN ansatz that can be paired with the classical ConvFilter.

The circuit reproduces the convolution‑pooling sequence from the original
QCNN implementation, but is wrapped in an EstimatorQNN so it can be used
together with the classical pre‑processing defined in QCNNGen088Model.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

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

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    idx = 0
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.compose(_conv_circuit(params[idx:idx+3]), [q1, q2], inplace=True)
        qc.barrier()
        idx += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.compose(_conv_circuit(params[idx:idx+3]), [q1, q2], inplace=True)
        qc.barrier()
        idx += 3
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

def _pool_layer(sources, sinks, param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for src, snk in zip(sources, sinks):
        qc.compose(_pool_circuit(params[idx:idx+3]), [src, snk], inplace=True)
        qc.barrier()
        idx += 3
    return qc

def QCNNGen088QNN() -> EstimatorQNN:
    estimator = Estimator()

    # Feature map that encodes the raw 8‑qubit input
    feature_map = ZFeatureMap(8).decompose()

    # Ansatz construction
    ansatz = QuantumCircuit(8, name="Ansatz")

    # Layer 1: convolution + pooling
    ansatz.compose(_conv_layer(8, "c1"), inplace=True)
    ansatz.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)

    # Layer 2: convolution + pooling
    ansatz.compose(_conv_layer(4, "c2"), inplace=True)
    ansatz.compose(_pool_layer([0, 1], [2, 3], "p2"), inplace=True)

    # Layer 3: convolution + pooling
    ansatz.compose(_conv_layer(2, "c3"), inplace=True)
    ansatz.compose(_pool_layer([0], [1], "p3"), inplace=True)

    # Full circuit
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

__all__ = ["QCNNGen088QNN"]
