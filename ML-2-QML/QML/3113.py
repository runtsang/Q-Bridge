"""Quantum QCNN with integrated quantum convolution filter.

This module extends the original QCNN implementation by adding a
quantum convolution filter (QuanvCircuit) before each convolutional
layer. The filter is a small random circuit that depends on the
classical input and introduces additional variational parameters.
The resulting hybrid circuit is then used as the ansatz for an
EstimatorQNN.

Public API:
- QCNNHybrid() returns an EstimatorQNN instance.
- QCNN() is an alias for compatibility.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.random import random_circuit

algorithm_globals.random_seed = 12345
estimator = Estimator()


def filter_circuit(params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
    """
    A small parameterised quantum filter that applies an RX rotation to
    each qubit followed by a shallow random circuit.  The filter is
    inspired by the quantum convolution filter in Conv.py but is
    unitary and suitable for inclusion in a variational ansatz.
    """
    qc = QuantumCircuit(len(qubits))
    for i, q in enumerate(qubits):
        qc.rx(params[i], q)
    qc += random_circuit(len(qubits), 2)
    return qc


def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Two-qubit convolution subcircuit from the original QCNN implementation.
    """
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """
    Convolutional layer that applies the conv_circuit on each adjacent pair of qubits.
    """
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Two-qubit pooling subcircuit from the original QCNN implementation.
    """
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target


def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """
    Pooling layer that applies the pool_circuit to each source-sink pair.
    """
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    pool_params = ParameterVector(f"{param_prefix}_p", length=num_qubits // 2 * 3)
    pool_idx = 0
    for src, snk in zip(sources, sinks):
        qc.append(pool_circuit(pool_params[pool_idx:pool_idx + 3]), [src, snk])
        pool_idx += 3
        qc.barrier()

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


def QCNNHybrid() -> EstimatorQNN:
    """
    Construct a QCNN ansatz that incorporates the quantum filter
    as a separate feature map.  The circuit is used to build
    an EstimatorQNN with a single Pauli‑Z observable on qubit 0.
    """
    # Feature map
    feature_map = ZFeatureMap(8)

    # Quantum filter applied as a feature map
    filter_params = ParameterVector("f", length=8)
    filter_layer_circuit = filter_circuit(filter_params, list(range(8)))

    # Ansatz construction
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First Convolutional Layer
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)

    # First Pooling Layer
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Second Convolutional Layer
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)

    # Second Pooling Layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third Convolutional Layer
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)

    # Third Pooling Layer
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Combine feature map, quantum filter and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(filter_layer_circuit, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters + list(filter_params),
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


def QCNN() -> EstimatorQNN:
    """Compatibility alias for the original QCNN factory."""
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "QCNN"]
