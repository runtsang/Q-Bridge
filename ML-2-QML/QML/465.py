"""Quantum QCNN with depthwise separable ansatz and variational loss."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def _depthwise_conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """
    A depthwise separable convolutional block that applies a single‑qubit
    rotation followed by a CNOT to its neighbor.
    Parameters are shared across qubits to reduce parameter count.
    """
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 2)
    for i in range(num_qubits):
        qc.ry(params[2 * i], i)
        qc.rz(params[2 * i + 1], i)
        qc.cx(i, (i + 1) % num_qubits)
    return qc

def _pooling_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """
    A pooling block that entangles qubits and applies a rotation.
    """
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits)
    for i in range(num_qubits):
        qc.cx(i, (i + 1) % num_qubits)
        qc.rz(params[i], (i + 1) % num_qubits)
    return qc

def QCNNAdvanced() -> EstimatorQNN:
    """
    Factory returning a variational QCNN with a depthwise separable ansatz.
    The circuit comprises:
    * A ZFeatureMap for encoding the 8‑dimensional input.
    * Three depthwise convolution layers interleaved with pooling layers.
    * A final measurement on the first qubit only.
    """
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Feature map
    feature_map = ZFeatureMap(8)

    # Build ansatz
    ansatz = QuantumCircuit(8, name="QCNNAdvAnsatz")
    ansatz.compose(_depthwise_conv_layer(8, "c1"), inplace=True)
    ansatz.compose(_pooling_layer(8, "p1"), inplace=True)
    ansatz.compose(_depthwise_conv_layer(8, "c2"), inplace=True)
    ansatz.compose(_pooling_layer(8, "p2"), inplace=True)
    ansatz.compose(_depthwise_conv_layer(8, "c3"), inplace=True)
    ansatz.compose(_pooling_layer(8, "p3"), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Observable: measurement on first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNNAdvanced"]
